"""[Opus 5.5] Lightweight temporal fusion on Mobile-CDNet (author BaseNet, models/model.py, unchanged).
A : LiteTF  - TFFM-style cascade of dilated convs (d = 7, 5, 3, 1) on the difference features d2..d5,
              but depthwise-separable and at Mobile-CDNet's own channel widths (24, 32, 96, 96); no NAM needed.
B : TemporalPair - besides |x1 - x2|, feed x1 + x2 ({|a-b|, a+b} <-> {max, min}: keeps both dates' content,
              invariant to pre/post order); fused by 1x1 conv, added back as a residual whose BN weight starts at 0
              (so A+B == A at initialisation).
C : apply A and B only at levels 3-5 (1/8 - 1/32); levels 1-2 untouched.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .model import BaseNet as MobileCDNet


def dws(c, d):
    return nn.Sequential(nn.Conv2d(c, c, 3, padding=d, dilation=d, groups=c, bias=False),
                         nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c))


class LiteTF(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.branch = nn.ModuleList([dws(c, d) for d in (7, 5, 3, 1)])
        self.skip = nn.ModuleList([nn.Conv2d(c, c, 1) for _ in range(4)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.branch[0](x)
        for k in range(1, 4):
            y = self.branch[k](self.relu(self.skip[k - 1](x) + y))
        return self.relu(self.skip[3](x) + y)


class TemporalPair(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pw = nn.Conv2d(2 * c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        nn.init.zeros_(self.bn.weight)

    def forward(self, a, b, d):
        return d + self.bn(self.pw(torch.cat([d, a + b], 1)))


class LiteTFNet(nn.Module):
    CH = [16, 24, 32, 96, 320]          # backbone levels 1..5
    DCH = {2: 24, 3: 32, 4: 96, 5: 96}  # channels of d2..d5 inside the author decoder

    def __init__(self, use_A=True, use_B=False, levels=(2, 3, 4, 5)):
        super().__init__()
        base = MobileCDNet(3, 1)
        self.backbone, self.swa = base.backbone, base.swa
        self.A = nn.ModuleDict({str(l): LiteTF(self.DCH[l]) for l in levels}) if use_A else nn.ModuleDict()
        self.B = nn.ModuleDict({str(l): TemporalPair(self.CH[l - 1]) for l in levels}) if use_B else nn.ModuleDict()

    def forward(self, x1, x2):
        f1, f2, s = self.backbone(x1), self.backbone(x2), self.swa
        c = []
        for l in range(1, 6):
            a, b = f1[l - 1], f2[l - 1]
            d = torch.abs(a - b)
            if str(l) in self.B:
                d = self.B[str(l)](a, b, d)
            c.append(d)
        c1, c2, c3, c4, c5 = c
        d1 = s.conv2d1(s.downsample(c1))
        d = {2: s.conv2d2(c2), 3: s.conv2d3(c3), 4: s.conv2d4(c4), 5: s.DoubleConv5(c5)}
        for l in (2, 3, 4, 5):
            if str(l) in self.A:
                d[l] = self.A[str(l)](d[l])
        f4 = s.cat4(torch.cat([d[4], F.interpolate(d[5], scale_factor=(2, 2), mode='bilinear')], 1))
        f3 = s.cat3(torch.cat([d[3], F.interpolate(f4, scale_factor=(2, 2), mode='bilinear')], 1))
        s2 = s.cat2(torch.cat([F.interpolate(f3, scale_factor=(2, 2), mode='bilinear'), d[2], d1], 1))
        return torch.sigmoid(F.interpolate(s.cls(s2), scale_factor=(4, 4), mode='bilinear'))


def build(variant):
    return {'litetf_A':   lambda: LiteTFNet(True, False, (2, 3, 4, 5)),
            'litetf_AB':  lambda: LiteTFNet(True, True, (2, 3, 4, 5)),
            'litetf_ABC': lambda: LiteTFNet(True, True, (3, 4, 5)),
            'litetf_none': lambda: LiteTFNet(False, False)}[variant]()
