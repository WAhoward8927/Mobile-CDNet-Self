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


class ResidualSelectiveFusion(nn.Module):
    """Recall-safe top-down selective fusion with an intact author path."""

    def __init__(self, high_channels, local_channels, out_channels):
        super().__init__()
        total_channels = high_channels + local_channels
        self.gate = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, kernel_size=3, padding=1,
                      groups=total_channels, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels, 2, kernel_size=1, bias=True)
        )
        self.correction = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.beta = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, high, local, base):
        joint = torch.cat([high, local], dim=1)
        gates = torch.sigmoid(self.gate(joint))
        high_gate = gates[:, 0:1]
        local_gate = gates[:, 1:2]
        correction = self.correction(
            torch.cat([high_gate * high, local_gate * local], dim=1)
        )
        return base + torch.tanh(self.beta) * correction


class NeighborFeatureAggregation(nn.Module):
    def __init__(self, in_d):
        super(NeighborFeatureAggregation, self).__init__()
        self.in_d = in_d
        self.downsample = nn.MaxPool2d(stride=2, kernel_size=2)
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(self.in_d[0], self.in_d[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d[1]), nn.ReLU(inplace=True))
        self.conv2d2 = nn.Sequential(
            nn.Conv2d(self.in_d[1], self.in_d[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d[1]), nn.ReLU(inplace=True))
        self.conv2d3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.in_d[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d[2]), nn.ReLU(inplace=True))
        self.conv2d4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.in_d[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d[3]), nn.ReLU(inplace=True))
        self.DoubleConv5 = DoubleConv(self.in_d[4], self.in_d[3])

        self.cat4 = DoubleConv(2 * self.in_d[3], self.in_d[2], self.in_d[3])
        self.cat3 = DoubleConv(2 * self.in_d[2], self.in_d[1], self.in_d[2])
        self.cat2 = DoubleConv(3 * self.in_d[1], self.in_d[1], 2 * self.in_d[1])

        # Preserve the complete author-module initialization order through cls.
        self.cls = nn.Conv2d(self.in_d[1], 1, kernel_size=3, stride=1, padding=1)
        # B1R-only modules are appended after the complete author path.
        self.select4 = ResidualSelectiveFusion(self.in_d[3], self.in_d[3], self.in_d[2])
        self.select3 = ResidualSelectiveFusion(self.in_d[2], self.in_d[2], self.in_d[1])
        self.select2 = ResidualSelectiveFusion(self.in_d[1], 2 * self.in_d[1], self.in_d[1])

    def gate_scales(self):
        return {
            'f4': float(torch.tanh(self.select4.beta).detach().cpu()),
            'f3': float(torch.tanh(self.select3.beta).detach().cpu()),
            's2': float(torch.tanh(self.select2.beta).detach().cpu())
        }

    def forward(self, x1_1, x1_2, x1_3, x1_4, x1_5,
                x2_1, x2_2, x2_3, x2_4, x2_5):
        c1 = torch.abs(x1_1 - x2_1)
        c2 = torch.abs(x1_2 - x2_2)
        c3 = torch.abs(x1_3 - x2_3)
        c4 = torch.abs(x1_4 - x2_4)
        c5 = torch.abs(x1_5 - x2_5)

        d1 = self.conv2d1(self.downsample(c1))
        d2 = self.conv2d2(c2)
        d3 = self.conv2d3(c3)
        d4 = self.conv2d4(c4)
        d5 = self.DoubleConv5(c5)

        e4 = F.interpolate(d5, scale_factor=(2, 2), mode='bilinear')
        f4_base = self.cat4(torch.cat([d4, e4], dim=1))
        f4 = self.select4(e4, d4, f4_base)

        e3 = F.interpolate(f4, scale_factor=(2, 2), mode='bilinear')
        f3_base = self.cat3(torch.cat([d3, e3], dim=1))
        f3 = self.select3(e3, d3, f3_base)

        e2 = F.interpolate(f3, scale_factor=(2, 2), mode='bilinear')
        local2 = torch.cat([d2, d1], dim=1)
        s2_base = self.cat2(torch.cat([e2, d2, d1], dim=1))
        s2 = self.select2(e2, local2, s2_base)
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
        mask = self.swa(x1_1, x1_2, x1_3, x1_4, x1_5,
                        x2_1, x2_2, x2_3, x2_4, x2_5)
        mask_p2 = F.interpolate(mask, scale_factor=(4, 4), mode='bilinear')
        return torch.sigmoid(mask_p2)
