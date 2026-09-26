import torch
from torch import nn
from torch.nn import functional as F
from .model import BaseNet


class HalfZeroCDNet(nn.Module):
    """Author decoder at quarter scale, then feature fusion at half scale.

    The added c1 slot is zero; the author's original d1 path still uses c1.
    """
    def __init__(self):
        super().__init__()
        self.base = BaseNet(3, 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(40, 40, 3, padding=1, groups=40, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.Conv2d(40, 24, 1, bias=False),
            nn.BatchNorm2d(24),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        self.half_cls = nn.Conv2d(24, 1, 3, padding=1)
        self.half_cls.load_state_dict(self.base.swa.cls.state_dict(), strict=True)
        # The original quarter-scale classifier is bypassed by this forward path.
        for parameter in self.base.swa.cls.parameters():
            parameter.requires_grad_(False)

    def forward(self, a, b):
        fa = self.base.backbone(a)
        fb = self.base.backbone(b)
        c1, c2, c3, c4, c5 = [torch.abs(x - y) for x, y in zip(fa, fb)]
        s = self.base.swa
        d1 = s.conv2d1(s.downsample(c1))
        d2 = s.conv2d2(c2)
        d3 = s.conv2d3(c3)
        d4 = s.conv2d4(c4)
        d5 = s.DoubleConv5(c5)
        f4 = s.cat4(torch.cat([d4, F.interpolate(d5, scale_factor=2, mode='bilinear')], dim=1))
        f3 = s.cat3(torch.cat([d3, F.interpolate(f4, scale_factor=2, mode='bilinear')], dim=1))
        s2 = s.cat2(torch.cat([F.interpolate(f3, scale_factor=2, mode='bilinear'), d2, d1], dim=1))
        up = F.interpolate(s2, size=c1.shape[-2:], mode='bilinear')
        detail = torch.zeros_like(c1)
        half_feature = up + self.fusion(torch.cat([up, detail], dim=1))
        half_logit = self.half_cls(half_feature)
        return torch.sigmoid(F.interpolate(half_logit, size=a.shape[-2:], mode='bilinear'))
