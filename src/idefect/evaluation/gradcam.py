"""Grad-CAM: where in the image did a CNN look?

Grad-CAM (Selvaraju et al., 2017) takes the last convolutional feature maps A
(before global pooling) and the gradient of the chosen class score y with
respect to them. Each map is weighted by its average gradient,

    w_k = mean_{i,j} dy / dA_k(i, j),    CAM = ReLU(sum_k w_k A_k),

and the result is upsampled to the input size. Bright regions are those that
raised the class score. The hook sits on the input of the network's global
pooling, which every architecture in :mod:`idefect.models.registry` has.

Grad-CAM shows where evidence was found, not why; at the coarse final
resolution (8x8 to 16x16 cells for a 128 px input) it localizes regions, not
single pixels.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _target(model: nn.Module) -> Tuple[str, nn.Module]:
    """("pre", m): capture the input of module m; ("out", m): its output."""
    from idefect.models.transfer import TransferNet

    if isinstance(model, TransferNet):
        net = model.net
        if hasattr(net, "avgpool"):          # torchvision ResNet
            return "pre", net.avgpool
        return "out", net.features           # DenseNet and MobileNetV2 pool functionally
    if hasattr(model, "pool"):
        return "pre", model.pool
    raise ValueError(f"no Grad-CAM target layer known for {type(model).__name__}")


class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.kind, self.layer = _target(model)
        self._acts: Optional[torch.Tensor] = None

    def _keep(self, t: torch.Tensor) -> None:
        t.retain_grad()
        self._acts = t

    def __call__(self, images: torch.Tensor, class_idx: Optional[List[int]] = None):
        """``images``: (N, 1, H, W). Returns ``(cams, probs)``: CAMs (N, H, W)
        in [0, 1] for ``class_idx`` (default: the predicted class) and the
        softmax probabilities (N, C)."""
        if self.kind == "pre":
            h = self.layer.register_forward_pre_hook(lambda m, inp: self._keep(inp[0]))
        else:
            h = self.layer.register_forward_hook(lambda m, inp, out: self._keep(out))
        was_training = self.model.training
        self.model.eval()
        try:
            x = images.clone().requires_grad_(True)   # graph exists even for frozen backbones
            with torch.enable_grad():
                logits = self.model(x)
                idx = logits.argmax(1) if class_idx is None else torch.as_tensor(class_idx, device=x.device)
                self.model.zero_grad(set_to_none=True)
                logits.gather(1, idx.view(-1, 1)).sum().backward()
            acts, grads = self._acts.detach(), self._acts.grad.detach()
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * acts).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            flat = cam.flatten(1)
            lo, hi = flat.min(1)[0].view(-1, 1, 1), flat.max(1)[0].view(-1, 1, 1)
            cam = (cam - lo) / (hi - lo).clamp_min(1e-8)
            probs = torch.softmax(logits.detach(), 1)
        finally:
            h.remove()
            self.model.train(was_training)
        return cam.cpu().numpy(), probs.cpu().numpy()


def cam_energy_inside(cam: np.ndarray, mask: np.ndarray) -> float:
    """Share of a CAM's total mass that falls inside a binary mask."""
    total = float(cam.sum())
    return float(cam[mask > 0].sum()) / total if total > 0 else 0.0
