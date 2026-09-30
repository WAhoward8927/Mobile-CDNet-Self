"""[Opus 5.5] A2Net component ablation (official A2Net code in models_a2net/model.py, unchanged).
variant 'full'   : official A2Net = MobileNetV2 + NAM (swa) + PCIM/TFFM (tfm) + SAM decoder with 4-scale deep supervision
variant 'noTFFM' : TFFM removed -> plain |f1 - f2| on the NAM features (64 ch, same shape the decoder expects)
variant 'noSAM'  : SAM removed from the decoder (no attention re-weighting, no side outputs / deep supervision);
                   the top-down path (conv_p4, conv_p3, conv_p2, cls) is kept exactly as in A2Net
forward returns a tuple of change maps (sigmoid, full resolution); element 0 is the final map, the rest are side outputs.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .model import BaseNet as A2Base


class A2NetAblation(A2Base):
    def __init__(self, variant='full'):
        super().__init__(3, 1)
        assert variant in ('full', 'noTFFM', 'noSAM'), variant
        self.variant = variant
        if variant == 'noTFFM':
            del self.tfm
        if variant == 'noSAM':
            del self.decoder.sam_p5, self.decoder.sam_p4, self.decoder.sam_p3

    def forward(self, x1, x2):
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)
        a = self.swa(x1_2, x1_3, x1_4, x1_5)
        b = self.swa(x2_2, x2_3, x2_4, x2_5)
        if self.variant == 'noTFFM':
            c2, c3, c4, c5 = [torch.abs(p - q) for p, q in zip(a, b)]
        else:
            c2, c3, c4, c5 = self.tfm(*a, *b)
        d = self.decoder
        if self.variant == 'noSAM':
            p5 = c5
            p4 = d.conv_p4(c4 + F.interpolate(p5, scale_factor=(2, 2), mode='bilinear'))
            p3 = d.conv_p3(c3 + F.interpolate(p4, scale_factor=(2, 2), mode='bilinear'))
            p2 = d.conv_p2(c2 + F.interpolate(p3, scale_factor=(2, 2), mode='bilinear'))
            return (torch.sigmoid(F.interpolate(d.cls(p2), scale_factor=(4, 4), mode='bilinear')),)
        _, _, _, _, m2, m3, m4, m5 = d(c2, c3, c4, c5)
        return tuple(torch.sigmoid(F.interpolate(m, scale_factor=(s, s), mode='bilinear')) for m, s in ((m2, 4), (m3, 8), (m4, 16), (m5, 32)))
