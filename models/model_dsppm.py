# [Opus 5.5] Author Mobile-CDNet + LightPPM(c5) + deep supervision(f4,f3)
import torch
from torch import nn
from torch.nn import functional as F
from .model import BaseNet


class LightPPM(nn.Module):
    """Pyramid pooling on c5 (1/32 scale). Residual with zero-init output BN,
    so at initialization the network is exactly the author model."""
    def __init__(self, in_ch=320, mid=64, bins=(1, 2, 4)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(b), nn.Conv2d(in_ch, mid, 1, bias=False),
                          nn.BatchNorm2d(mid), nn.ReLU(inplace=True)) for b in bins])
        self.project = nn.Sequential(
            nn.Conv2d(mid * len(bins), in_ch, 1, bias=False), nn.BatchNorm2d(in_ch))
        nn.init.zeros_(self.project[1].weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        ctx = [F.interpolate(s(x), size=(h, w), mode='bilinear', align_corners=False) for s in self.stages]
        return x + self.project(torch.cat(ctx, 1))


class DSPPMCDNet(nn.Module):
    """Author Mobile-CDNet + optional LightPPM on c5 + deep-supervision heads on f4 (1/16) and f3 (1/8).
    Aux heads are used only in training; inference output is identical in form to the author model."""
    def __init__(self, use_ppm=True, use_ds=True):
        super().__init__()
        self.base = BaseNet(3, 1)
        self.use_ppm, self.use_ds = use_ppm, use_ds
        self.ppm = LightPPM(320) if use_ppm else None
        if use_ds:
            self.aux4 = nn.Conv2d(32, 1, 3, padding=1)   # f4: 32 ch @ 1/16
            self.aux3 = nn.Conv2d(24, 1, 3, padding=1)   # f3: 24 ch @ 1/8

    def forward(self, a, b):
        fa = self.base.backbone(a)
        fb = self.base.backbone(b)
        c1, c2, c3, c4, c5 = [torch.abs(x - y) for x, y in zip(fa, fb)]
        if self.use_ppm:
            c5 = self.ppm(c5)
        s = self.base.swa
        d1 = s.conv2d1(s.downsample(c1))
        d2 = s.conv2d2(c2)
        d3 = s.conv2d3(c3)
        d4 = s.conv2d4(c4)
        d5 = s.DoubleConv5(c5)
        f4 = s.cat4(torch.cat([d4, F.interpolate(d5, scale_factor=2, mode='bilinear')], 1))
        f3 = s.cat3(torch.cat([d3, F.interpolate(f4, scale_factor=2, mode='bilinear')], 1))
        s2 = s.cat2(torch.cat([F.interpolate(f3, scale_factor=2, mode='bilinear'), d2, d1], 1))
        size = a.shape[-2:]
        out = torch.sigmoid(F.interpolate(s.cls(s2), size=size, mode='bilinear'))
        if self.training and self.use_ds:
            o4 = torch.sigmoid(F.interpolate(self.aux4(f4), size=size, mode='bilinear'))
            o3 = torch.sigmoid(F.interpolate(self.aux3(f3), size=size, mode='bilinear'))
            return out, [o4, o3]
        return out
