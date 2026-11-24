"""One place that maps an architecture name to its network.

The trainer, the predictor, the app and every study build CNNs through
:func:`build_cnn`, so a checkpoint only needs to store the architecture name
(and, for ImageNet backbones, the fine-tuning mode) to be rebuilt exactly.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from idefect.models.custom_cnn import CustomCNN
from idefect.models.densenet import DenseNet
from idefect.models.mobilenet import MobileNetV2
from idefect.models.resnet import ResNet
from idefect.models.transfer import BACKBONES, FINETUNE_MODES, TransferNet
from idefect.models.vgg_style import VGGStyle

# name -> (display name, constructor(num_classes, in_channels, base_channels, dropout))
_ARCHS: Dict[str, tuple] = {
    "custom_cnn": ("Custom CNN", lambda n, i, c, d: CustomCNN(n, i, base_channels=c, dropout=d)),
    "vgg": ("VGG-style CNN", lambda n, i, c, d: VGGStyle(n, i, base_channels=c, dropout=d)),
    "resnet": ("ResNet-18 (scratch)", lambda n, i, c, d: ResNet(n, i, base_channels=c, dropout=d)),
    "densenet": ("DenseNet-BC (scratch)", lambda n, i, c, d: DenseNet(n, i, dropout=d)),
    "mobilenet": ("MobileNetV2 (scratch)", lambda n, i, c, d: MobileNetV2(n, i, dropout=d)),
}

ARCH_ORDER = list(_ARCHS) + list(BACKBONES)


def available() -> list:
    return list(_ARCHS) + list(BACKBONES)


def is_transfer(arch: str) -> bool:
    """True for ImageNet-pretrained backbones (they take a fine-tuning mode)."""
    return arch in BACKBONES


def artifact_stem(arch: str, finetune: str = "full") -> str:
    """File stem for an architecture's artifacts: the name, plus the
    fine-tuning mode for ImageNet backbones (``resnet18_imagenet_frozen``)."""
    return f"{arch}_{finetune}" if is_transfer(arch) else arch


def display_name(arch: str, finetune: str = "full") -> str:
    if arch in _ARCHS:
        return _ARCHS[arch][0]
    if arch in BACKBONES:
        return f"{BACKBONES[arch][0][:-1]}, {finetune})"
    return arch


def build_cnn(arch: str, num_classes: int, in_channels: int = 1, base_channels: int = 32,
              dropout: float = 0.3, finetune: str = "full", pretrained: bool = False) -> nn.Module:
    """Build architecture ``arch``.

    ``base_channels`` sets the width of the custom CNN, the VGG-style network
    and the ResNet; DenseNet and MobileNet use their published widths. For
    ImageNet backbones ``finetune`` picks frozen, partial or full fine-tuning
    and ``pretrained`` loads the ImageNet weights (training only; a saved
    checkpoint is loaded over a randomly initialized network, no download).
    """
    if arch in BACKBONES:
        if finetune not in FINETUNE_MODES:
            raise ValueError(f"finetune must be one of {FINETUNE_MODES}")
        return TransferNet(arch, num_classes, finetune=finetune, pretrained=pretrained, dropout=dropout)
    if arch not in _ARCHS:
        raise ValueError(f"unknown architecture '{arch}'; choose from {available()}")
    builder: Callable = _ARCHS[arch][1]
    return builder(num_classes, in_channels, base_channels, dropout)
