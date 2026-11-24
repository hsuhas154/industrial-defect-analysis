"""DenseNet-BC implemented from scratch.

In a dense block every layer receives the feature maps of *all* earlier
layers (concatenated) and adds ``growth`` new maps of its own. Features are
reused instead of re-learned, so the network stays narrow and parameter
efficient, and every layer has a short path to the loss.

DenseNet-BC adds a 1x1 "bottleneck" convolution (to 4 x growth channels)
before each 3x3 convolution, and transition layers that compress the channel
count by half (1x1 conv) and halve the resolution (2x2 average pooling).

Layout for 128x128 grayscale input:

    stem:        Conv3x3(2 x growth), BN, ReLU, MaxPool    (128 -> 64)
    dense block: 6 layers  -> transition                   (64 -> 32)
    dense block: 12 layers -> transition                   (32 -> 16)
    dense block: 16 layers                                 (16)
    BN, ReLU, global average pooling -> Dropout -> Linear
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    """BN-ReLU-Conv1x1 (bottleneck) then BN-ReLU-Conv3x3, output concatenated."""

    def __init__(self, in_ch: int, growth: int, bottleneck: int = 4):
        super().__init__()
        mid = bottleneck * growth
        self.body = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True), nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True), nn.Conv2d(mid, growth, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.body(x)], dim=1)


class Transition(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
                         nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.AvgPool2d(2))


class DenseNet(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, growth: int = 16,
                 block_layers: Sequence[int] = (6, 12, 16), compression: float = 0.5,
                 dropout: float = 0.3):
        super().__init__()
        ch = 2 * growth
        layers = [nn.Conv2d(in_channels, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
                  nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        for i, n in enumerate(block_layers):
            for _ in range(n):
                layers.append(DenseLayer(ch, growth))
                ch += growth
            if i < len(block_layers) - 1:
                out = int(ch * compression)
                layers.append(Transition(ch, out))
                ch = out
        layers += [nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(ch, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))
