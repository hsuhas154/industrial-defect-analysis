"""One place that maps an architecture name to its network.

The trainer, the predictor, the app and every study build CNNs through
:func:`build_cnn`, so a checkpoint only needs to store the architecture name
to be rebuilt exactly.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from idefect.models.custom_cnn import CustomCNN
from idefect.models.densenet import DenseNet
from idefect.models.mobilenet import MobileNetV2
from idefect.models.resnet import ResNet
from idefect.models.vgg_style import VGGStyle

# name -> (display name, constructor(num_classes, in_channels, base_channels, dropout))
_ARCHS: Dict[str, tuple] = {
    "custom_cnn": ("Custom CNN", lambda n, i, c, d: CustomCNN(n, i, base_channels=c, dropout=d)),
    "vgg": ("VGG-style CNN", lambda n, i, c, d: VGGStyle(n, i, base_channels=c, dropout=d)),
    "resnet": ("ResNet-18 (scratch)", lambda n, i, c, d: ResNet(n, i, base_channels=c, dropout=d)),
    "densenet": ("DenseNet-BC (scratch)", lambda n, i, c, d: DenseNet(n, i, dropout=d)),
    "mobilenet": ("MobileNetV2 (scratch)", lambda n, i, c, d: MobileNetV2(n, i, dropout=d)),
}

ARCH_ORDER = list(_ARCHS)


def available() -> list:
    return list(_ARCHS)


def display_name(arch: str) -> str:
    return _ARCHS[arch][0] if arch in _ARCHS else arch


def build_cnn(arch: str, num_classes: int, in_channels: int = 1, base_channels: int = 32,
              dropout: float = 0.3) -> nn.Module:
    """Build architecture ``arch``. ``base_channels`` sets the width of the
    custom CNN, the VGG-style network and the ResNet; DenseNet and MobileNet
    use their published widths."""
    if arch not in _ARCHS:
        raise ValueError(f"unknown architecture '{arch}'; choose from {available()}")
    builder: Callable = _ARCHS[arch][1]
    return builder(num_classes, in_channels, base_channels, dropout)
