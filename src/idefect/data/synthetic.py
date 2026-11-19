"""Generate a small synthetic, NEU-style dataset.

This is a smoke-test stand-in so the full pipeline and the unit tests can run
without downloading the real NEU database. Each class gets a distinct, simple
procedural texture. It is **not** real defect data and must not be used to
report results.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from idefect import NEU_CLASSES


def _texture_for_class(cls_idx: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Produce a distinctive grayscale texture per class index."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = rng.normal(128, 12, size=(size, size)).astype(np.float32)

    if cls_idx == 0:  # crazing: fine cross-hatch
        pattern = 40 * (np.sin(xx / 2.0) * np.sin(yy / 2.0))
    elif cls_idx == 1:  # inclusion: random blobs
        pattern = np.zeros((size, size), np.float32)
        for _ in range(6):
            cx, cy = rng.integers(0, size, 2)
            r = int(rng.integers(4, 12))
            cv2.circle(pattern, (int(cx), int(cy)), r, float(rng.integers(40, 90)), -1)
    elif cls_idx == 2:  # patches: large soft regions
        pattern = 50 * np.sin((xx + yy) / 20.0)
    elif cls_idx == 3:  # pitted_surface: speckle
        pattern = rng.normal(0, 45, size=(size, size)).astype(np.float32)
    elif cls_idx == 4:  # rolled-in_scale: horizontal banding
        pattern = 45 * np.sin(yy / 4.0)
    else:  # scratches: diagonal streaks
        pattern = np.zeros((size, size), np.float32)
        for _ in range(5):
            x0 = int(rng.integers(0, size))
            cv2.line(pattern, (x0, 0), (x0 + int(rng.integers(-30, 30)), size - 1),
                     float(rng.integers(60, 120)), 1)

    img = np.clip(base + pattern, 0, 255).astype(np.uint8)
    return img


def generate_dummy_dataset(
    out_dir: str | Path,
    per_class: int = 40,
    size: int = 200,
    seed: int = 0,
) -> Path:
    """Create class sub-folders of synthetic images under ``out_dir``.

    Returns the output directory path.
    """
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    for cls_idx, cls in enumerate(NEU_CLASSES):
        cdir = out_dir / cls
        cdir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = _texture_for_class(cls_idx, size, rng)
            cv2.imwrite(str(cdir / f"{cls}_{i:03d}.png"), img)
    return out_dir
