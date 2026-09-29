"""[Opus 5.5] E1 / E2 modules for Mobile-CDNet, adapted from BiFA (Zhang et al., IEEE TGRS 2024, github.com/zmoka-zht/BiFA).
E1  ADFF  : DiffFlow alignment before |difference| at c2, c3, c4 (1/4, 1/8, 1/16). BiFA Table VI: stages 1-3 best, adding the
            deepest stage hurts -> c1 (1/2) and c5 (1/32) keep plain |f1 - f2|. Official design: flowmlp per branch,
            3x3 conv on concat -> 2-ch flow, warp T1 (pre) onto T2 (post; labels are aligned to post), |warp(f1') - f2'|.
            Changes vs official: flow expressed in pixels (official divides by size -> half-pixel units), flow conv zero-init
            (starts as identity warp), explicit align_corners.
E2  BI    : bitemporal channel cross-attention inserted after every MobileNetV2 stage (c1..c5), shared weights,
            f1 <- f1 + BI(f1 | f2), f2 <- f2 + BI(f2 | f1); the interacted features feed the next stage (as in BiFA).
            Official BI is inside SegFormer blocks; here it is a residual CNN plug-in: LayerNorm over channels,
            1x1 q / kv projections, channel-wise attention per head. Change vs official: q,k L2-normalised over the spatial
            axis with a learnable temperature (official uses head_dim^-0.5 on raw dot products over N = H*W tokens, which
            saturates the softmax at 128x128 maps); output projection zero-init -> the network starts exactly as Mobile-CDNet.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .model import BaseNet

# MobileNetV2.features indices whose outputs are c1..c5 (see MobileNetV2.forward: idx in [1, 3, 6, 13, 17])
STAGES = [(0, 2), (2, 4), (4, 7), (7, 14), (14, 18)]
CH = [16, 24, 32, 96, 320]


class FlowMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.enlarge = nn.Conv2d(c, 4 * c, 1)
        self.dw = nn.Conv2d(4 * c, 4 * c, 3, 1, 1, groups=c)
        self.shrink = nn.Conv2d(4 * c, c, 1)

    def forward(self, x):
        return self.shrink(F.gelu(self.dw(self.enlarge(x))))


class DiffFlow(nn.Module):
    """ADFF: |warp(f1', flow) - f2'|, flow predicted from concat(f1', f2')."""
    def __init__(self, c):
        super().__init__()
        self.m1, self.m2 = FlowMLP(c), FlowMLP(c)
        self.flow = nn.Conv2d(2 * c, 2, 3, 1, 1, bias=False)
        nn.init.zeros_(self.flow.weight)
        self.last_flow = None

    def forward(self, f1, f2):
        a, b = self.m1(f1), self.m2(f2)
        flow = self.flow(torch.cat([a, b], 1))              # (B, 2, H, W), in feature pixels
        self.last_flow = flow.detach()
        B, _, H, W = a.shape
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=a.device, dtype=a.dtype),
                                torch.linspace(-1, 1, W, device=a.device, dtype=a.dtype), indexing='ij')
        grid = torch.stack([xs, ys], -1).unsqueeze(0).expand(B, -1, -1, -1)
        scale = torch.tensor([2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)], device=a.device, dtype=a.dtype)
        grid = grid + flow.permute(0, 2, 3, 1) * scale
        warped = F.grid_sample(a, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return torch.abs(warped - b)


class BI(nn.Module):
    """Bitemporal channel cross-attention (residual, zero-init output)."""
    def __init__(self, c, heads=8):
        super().__init__()
        self.h = heads if c % heads == 0 and c // heads >= 4 else max(1, c // 8)
        self.norm = nn.LayerNorm(c)
        self.q = nn.Conv2d(c, c, 1, bias=False)
        self.kv = nn.Conv2d(c, 2 * c, 1, bias=False)
        self.temp = nn.Parameter(torch.ones(self.h, 1, 1))
        self.proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def ln(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def attend(self, x, cond):
        B, C, H, W = x.shape
        q = self.q(self.ln(x)).reshape(B, self.h, C // self.h, H * W)
        k, v = self.kv(self.ln(cond)).reshape(B, 2, self.h, C // self.h, H * W).unbind(1)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temp).softmax(-1)      # (B, h, C/h, C/h)
        return x + self.proj((attn @ v).reshape(B, C, H, W))

    def forward(self, f1, f2):
        return self.attend(f1, f2), self.attend(f2, f1)


class MobileCDNetBiFA(nn.Module):
    """arch: 'adff' (E1), 'bi' (E2), 'bi_adff' (E1+E2). Decoder / head = author's NeighborFeatureAggregation, unchanged."""
    def __init__(self, arch='adff', bi_stages=(0, 1, 2, 3, 4), adff_stages=(1, 2, 3)):
        super().__init__()
        self.base = BaseNet(3, 1)
        self.use_bi, self.use_adff = 'bi' in arch, 'adff' in arch
        self.bi_stages, self.adff_stages = set(bi_stages), set(adff_stages)
        self.bi = nn.ModuleList([BI(c) for c in CH]) if self.use_bi else None
        self.adff = nn.ModuleList([DiffFlow(c) if i in self.adff_stages else nn.Identity() for i, c in enumerate(CH)]) if self.use_adff else None

    def forward(self, x1, x2):
        feats = self.base.backbone.features
        a, b, fa, fb = x1, x2, [], []
        for i, (s, e) in enumerate(STAGES):
            for j in range(s, e):
                a, b = feats[j](a), feats[j](b)
            if self.use_bi and i in self.bi_stages:
                a, b = self.bi[i](a, b)
            fa.append(a); fb.append(b)
        c = [self.adff[i](fa[i], fb[i]) if (self.use_adff and i in self.adff_stages) else torch.abs(fa[i] - fb[i]) for i in range(5)]
        s = self.base.swa
        d1 = s.conv2d1(s.downsample(c[0])); d2 = s.conv2d2(c[1]); d3 = s.conv2d3(c[2]); d4 = s.conv2d4(c[3]); d5 = s.DoubleConv5(c[4])
        f4 = s.cat4(torch.cat([d4, F.interpolate(d5, scale_factor=(2, 2), mode='bilinear')], 1))
        f3 = s.cat3(torch.cat([d3, F.interpolate(f4, scale_factor=(2, 2), mode='bilinear')], 1))
        s2 = s.cat2(torch.cat([F.interpolate(f3, scale_factor=(2, 2), mode='bilinear'), d2, d1], 1))
        return torch.sigmoid(F.interpolate(s.cls(s2), scale_factor=(4, 4), mode='bilinear'))
