"""ImageNet-pretrained backbones for transfer learning (torchvision).

Question: does knowledge from 1.2 million natural photographs help on 1,260
grayscale steel images? Three fine-tuning depths are compared:

* ``frozen``:  the pretrained backbone is a fixed feature extractor; only a new
  linear head is trained. Its BatchNorm layers stay in evaluation mode, so not
  even their running statistics move.
* ``partial``: the last stage of the backbone is unfrozen as well.
* ``full``:    every layer is trained.

Unfrozen pretrained layers learn ``backbone_lr_mult`` times slower than the new
head (0.1 by default), the usual way to adapt pretrained features without
destroying them.

NEU images are grayscale in [0, 1]; :class:`GrayToImageNet` repeats the channel
three times and applies the ImageNet mean and standard deviation, inside the
model, so every caller (trainer, predictor, app, Grad-CAM) feeds it exactly
like the networks trained from scratch.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FINETUNE_MODES = ("frozen", "partial", "full")


class GrayToImageNet(nn.Module):
    """(N, 1, H, W) in [0, 1] -> (N, 3, H, W) normalized like ImageNet inputs."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return (x - self.mean) / self.std


def _torchvision(name: str, pretrained: bool):
    import torchvision.models as M

    weights = {"resnet18": M.ResNet18_Weights, "densenet121": M.DenseNet121_Weights,
               "mobilenet_v2": M.MobileNet_V2_Weights}[name].IMAGENET1K_V1 if pretrained else None
    return getattr(M, name)(weights=weights)


# backbone -> (display name, torchvision name, head attribute, last-stage attributes for "partial")
BACKBONES: Dict[str, tuple] = {
    "resnet18_imagenet": ("ResNet-18 (ImageNet)", "resnet18", "fc", ["layer4"]),
    "densenet121_imagenet": ("DenseNet-121 (ImageNet)", "densenet121", "classifier",
                             ["features.denseblock4", "features.norm5"]),
    "mobilenet_v2_imagenet": ("MobileNetV2 (ImageNet)", "mobilenet_v2", "classifier",
                              [f"features.{i}" for i in range(14, 19)]),
}


def _get(module: nn.Module, dotted: str) -> nn.Module:
    for part in dotted.split("."):
        module = getattr(module, part)
    return module


class TransferNet(nn.Module):
    def __init__(self, backbone: str, num_classes: int, finetune: str = "full",
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone '{backbone}'; choose from {list(BACKBONES)}")
        if finetune not in FINETUNE_MODES:
            raise ValueError(f"finetune must be one of {FINETUNE_MODES}")
        _, tv_name, head_attr, last_stage = BACKBONES[backbone]
        self.backbone_name, self.finetune = backbone, finetune
        self.adapter = GrayToImageNet()
        self.net = _torchvision(tv_name, pretrained)
        old_head = getattr(self.net, head_attr)
        in_features = (old_head[-1] if isinstance(old_head, nn.Sequential) else old_head).in_features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        setattr(self.net, head_attr, self.head)

        trainable: List[nn.Module] = []
        if finetune == "full":
            trainable = [m for name, m in self.net.named_children() if name != head_attr]
        elif finetune == "partial":
            trainable = [_get(self.net, a) for a in last_stage]
        # A plain list, not a ModuleList: these modules are already registered
        # inside self.net, and registering them twice would duplicate weights
        # in the state dict.
        self._unfrozen = list(trainable)
        for p in self.net.parameters():
            p.requires_grad = False
        for m in list(trainable) + [self.head]:
            for p in m.parameters():
                p.requires_grad = True

    def train(self, mode: bool = True):
        """Frozen pretrained layers always stay in evaluation mode, so their
        BatchNorm statistics remain the ImageNet ones."""
        super().train(mode)
        if mode and self.finetune != "full":
            self.net.eval()
            for m in list(self._unfrozen) + [self.head]:
                m.train()
        return self

    def param_groups(self, lr: float, backbone_lr_mult: float = 0.1) -> List[Dict]:
        """The new head learns at ``lr``, unfrozen pretrained layers at
        ``lr * backbone_lr_mult``."""
        head = list(self.head.parameters())
        head_ids = {id(p) for p in head}
        body = [p for p in self.net.parameters() if p.requires_grad and id(p) not in head_ids]
        groups = [{"params": head, "lr": lr}]
        if body:
            groups.append({"params": body, "lr": lr * backbone_lr_mult})
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.adapter(x))
