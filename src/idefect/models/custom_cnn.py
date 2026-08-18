"""A small custom CNN implemented from scratch in PyTorch.

The network learns a representation directly from pixels — no handcrafted
feature extraction — which is exactly the contrast Phase 1 is built to show
against the HOG + Decision Tree / SVM branch.

Architecture (deliberately small so it trains fast on a laptop GPU or CPU):

    Conv(1  -> c)   -> BN -> ReLU -> MaxPool
    Conv(c  -> 2c)  -> BN -> ReLU -> MaxPool
    Conv(2c -> 4c)  -> BN -> ReLU -> MaxPool
    Global Average Pool -> Dropout -> Linear(4c -> num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CustomCNN(nn.Module):
    """Compact 3-block CNN classifier."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            _conv_block(in_channels, c),
            _conv_block(c, 2 * c),
            _conv_block(2 * c, 4 * c),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> (N, 4c, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(4 * c, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
