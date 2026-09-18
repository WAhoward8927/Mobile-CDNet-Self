import torch
import torch.nn as nn
import torch.nn.functional as F
from . import MobileNetV2


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SymmetricPreDiffGate(nn.Module):
    """Shared temporal DWConv gate applied before the final absolute difference."""

    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        # Process both dates together so BN statistics and the gate are order-symmetric.
        encoded = self.bn(self.dw(torch.cat([x1, x2], dim=0)))
        e1, e2 = encoded.chunk(2, dim=0)
        gate = torch.sigmoid(torch.abs(e1 - e2))
        scale = 1.0 + torch.tanh(self.alpha) * gate
        return torch.abs(x1 * scale - x2 * scale)

    def scale(self):
        return float(torch.tanh(self.alpha).detach().cpu())


class NeighborFeatureAggregation(nn.Module):
    def __init__(self, in_d):
        super(NeighborFeatureAggregation, self).__init__()
        self.in_d = in_d
        self.downsample = nn.MaxPool2d(stride=2, kernel_size=2)

        self.conv2d1 = nn.Sequential(
            nn.Conv2d(self.in_d[0], self.in_d[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d[1]),
            nn.ReLU(inplace=True)
        )
        self.conv2d2 = nn.Sequential(
            nn.Conv2d(self.in_d[1], self.in_d[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d[1]),
            nn.ReLU(inplace=True)
        )
        self.conv2d3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.in_d[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d[2]),
            nn.ReLU(inplace=True)
        )
        self.conv2d4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.in_d[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d[3]),
            nn.ReLU(inplace=True)
        )
        self.DoubleConv5 = DoubleConv(self.in_d[4], self.in_d[3])

        self.cat4 = DoubleConv(2 * self.in_d[3], self.in_d[2], self.in_d[3])
        self.cat3 = DoubleConv(2 * self.in_d[2], self.in_d[1], self.in_d[2])
        self.cat2 = DoubleConv(3 * self.in_d[1], self.in_d[1], 2 * self.in_d[1])

        # Keep every author module, including cls, in the original initialization order.
        self.cls = nn.Conv2d(self.in_d[1], 1, kernel_size=3, stride=1, padding=1)

        # B2-only modules are initialized after the complete author path.
        self.pre_diff = nn.ModuleList([
            SymmetricPreDiffGate(channels) for channels in self.in_d
        ])

    def gate_scales(self):
        return {
            'c1': self.pre_diff[0].scale(),
            'c2': self.pre_diff[1].scale(),
            'c3': self.pre_diff[2].scale(),
            'c4': self.pre_diff[3].scale(),
            'c5': self.pre_diff[4].scale(),
        }

    def forward(self, x1_1, x1_2, x1_3, x1_4, x1_5,
                x2_1, x2_2, x2_3, x2_4, x2_5):
        x1_features = (x1_1, x1_2, x1_3, x1_4, x1_5)
        x2_features = (x2_1, x2_2, x2_3, x2_4, x2_5)
        c1, c2, c3, c4, c5 = [
            gate(a, b)
            for gate, a, b in zip(self.pre_diff, x1_features, x2_features)
        ]

        d1 = self.downsample(c1)
        d1 = self.conv2d1(d1)
        d2 = self.conv2d2(c2)
        d3 = self.conv2d3(c3)
        d4 = self.conv2d4(c4)
        d5 = self.DoubleConv5(c5)

        e4 = F.interpolate(d5, scale_factor=(2, 2), mode='bilinear')
        f4 = self.cat4(torch.cat([d4, e4], dim=1))
        e3 = F.interpolate(f4, scale_factor=(2, 2), mode='bilinear')
        f3 = self.cat3(torch.cat([d3, e3], dim=1))
        e2 = F.interpolate(f3, scale_factor=(2, 2), mode='bilinear')
        s2 = self.cat2(torch.cat([e2, d2, d1], dim=1))
        return self.cls(s2)


class BaseNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1):
        super(BaseNet, self).__init__()
        self.backbone = MobileNetV2.mobilenet_v2(pretrained=True)
        channels = [16, 24, 32, 96, 320]
        self.swa = NeighborFeatureAggregation(channels)

    def forward(self, x1, x2):
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)
        mask = self.swa(
            x1_1, x1_2, x1_3, x1_4, x1_5,
            x2_1, x2_2, x2_3, x2_4, x2_5
        )
        mask_p2 = F.interpolate(mask, scale_factor=(4, 4), mode='bilinear')
        return torch.sigmoid(mask_p2)
