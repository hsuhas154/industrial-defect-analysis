"""MobileNetV2 implemented from scratch.

MobileNet replaces a standard 3x3 convolution with a depthwise separable one:
a depthwise 3x3 convolution (one filter per channel, spatial filtering only)
followed by a pointwise 1x1 convolution (mixing channels). For C channels in
and out this costs about 9C + C^2 multiplications per pixel instead of 9C^2,
roughly 8 to 9 times fewer for typical widths.

MobileNetV2's inverted residual block expands the channels with a 1x1
convolution (factor t), filters depthwise in that wide space, then projects
back down with a *linear* 1x1 convolution (no ReLU, which would destroy
information in the narrow space). A residual connection joins input and
output when their shapes match.

Layout (the published MobileNetV2 table, width multiplier 1.0), for 128x128
grayscale input:

    stem: Conv3x3(32, stride 2), BN, ReLU6                      (128 -> 64)
    t, c, n, s:  1,16,1,1 | 6,24,2,2 | 6,32,3,2 | 6,64,4,2 |
                 6,96,3,1 | 6,160,3,2 | 6,320,1,1               (64 -> 4)
    Conv1x1(1280), BN, ReLU6, global average pooling -> Dropout -> Linear
"""

from __future__ import annotations

import torch
import torch.nn as nn

SETTINGS = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
            (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]


def _round(ch: float, divisor: int = 8) -> int:
    """Round a channel count to a multiple of ``divisor`` (as in the paper's code)."""
    v = max(divisor, int(ch + divisor / 2) // divisor * divisor)
    return v + divisor if v < 0.9 * ch else v


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand: int):
        super().__init__()
        mid = in_ch * expand
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU6(inplace=True)]
        layers += [
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),  # depthwise
            nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
            nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),            # linear projection
        ]
        self.body = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) if self.use_residual else self.body(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, width: float = 1.0,
                 dropout: float = 0.3):
        super().__init__()
        ch = _round(32 * width)
        layers = [nn.Conv2d(in_channels, ch, 3, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(ch), nn.ReLU6(inplace=True)]
        for t, c, n, s in SETTINGS:
            out = _round(c * width)
            for i in range(n):
                layers.append(InvertedResidual(ch, out, s if i == 0 else 1, t))
                ch = out
        last = _round(1280 * max(1.0, width))
        layers += [nn.Conv2d(ch, last, 1, bias=False), nn.BatchNorm2d(last), nn.ReLU6(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(last, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))
