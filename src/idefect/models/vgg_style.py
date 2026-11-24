"""VGG-style CNN implemented from scratch.

VGG's idea is depth through repetition: stacks of 3x3 convolutions (two 3x3
layers see a 5x5 region with fewer parameters and one more non-linearity than
a single 5x5), with max pooling halving the resolution between stages and the
width doubling. This is the VGG-11/13 pattern with BatchNorm, adapted to
128x128 grayscale input:

    stage 1:  [Conv3x3(c),  BN, ReLU] x 2  -> MaxPool   (128 -> 64)
    stage 2:  [Conv3x3(2c), BN, ReLU] x 2  -> MaxPool   (64 -> 32)
    stage 3:  [Conv3x3(4c), BN, ReLU] x 2  -> MaxPool   (32 -> 16)
    stage 4:  [Conv3x3(8c), BN, ReLU] x 2  -> MaxPool   (16 -> 8)
    Global average pooling -> Dropout -> Linear

The original VGG ends in two 4096-unit fully connected layers (about 120 of
its 138 million parameters). On 1,260 training images that head would only
memorize, so global average pooling replaces it, as in later networks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv(in_ch: int, out_ch: int) -> list:
    return [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]


class VGGStyle(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, base_channels: int = 32,
                 convs_per_stage: int = 2, dropout: float = 0.3):
        super().__init__()
        layers, ch = [], in_channels
        for width in (base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels):
            for _ in range(convs_per_stage):
                layers += _conv(ch, width)
                ch = width
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(ch, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))
