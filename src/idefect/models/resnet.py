"""ResNet (ResNet-18 layout) implemented from scratch.

A residual block learns a correction F(x) that is added to its input:
y = ReLU(F(x) + shortcut(x)). The identity path lets gradients reach early
layers unchanged, so much deeper networks train reliably, and a block that has
nothing useful to add can fall back to (almost) the identity.

Layout, adapted to 128x128 grayscale input (no 7x7 stem and no stem max pool,
which would discard fine texture at this small size):

    stem:     Conv3x3(c), BN, ReLU                         (128)
    stage 1:  BasicBlock(c)  x 2                           (128)
    stage 2:  BasicBlock(2c) x 2, first with stride 2      (64)
    stage 3:  BasicBlock(4c) x 2, first with stride 2      (32)
    stage 4:  BasicBlock(8c) x 2, first with stride 2      (16)
    Global average pooling -> Dropout -> Linear

BasicBlock: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN, plus a 1x1 projection
shortcut whenever the shape changes. The last BN of each block starts at zero,
so every block starts as the identity (a common, well-tested initialization).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, base_channels: int = 32,
                 blocks: Sequence[int] = (2, 2, 2, 2), dropout: float = 0.3):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        stages, ch = [], c
        for i, n in enumerate(blocks):
            width = c * 2 ** i
            for j in range(n):
                stages.append(BasicBlock(ch, width, stride=2 if (i > 0 and j == 0) else 1))
                ch = width
        self.layers = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(ch, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.layers(self.stem(x))))
