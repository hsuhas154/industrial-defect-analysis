"""Classical computer-vision preprocessing.

All functions operate on single-channel (grayscale) images represented as NumPy
arrays. The pipeline is intentionally explicit and configurable so we can study
whether each step helps or hurts a downstream model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from idefect.config import PreprocessingConfig


def load_grayscale(path: str | Path) -> np.ndarray:
    """Load an image from disk as a uint8 grayscale array (H, W)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize to (size, size). Uses area interpolation for downscaling."""
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def apply_clahe(img: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalisation on a uint8 image."""
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid)
    )
    return clahe.apply(img)


def denoise(img: np.ndarray, method: str, ksize: int) -> np.ndarray:
    """Optional denoising. ``method`` is one of none|gaussian|median."""
    if method == "none":
        return img
    if method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if method == "median":
        return cv2.medianBlur(img, ksize)
    raise ValueError(f"unknown denoise method: {method}")


def preprocess_image(
    img: np.ndarray,
    cfg: PreprocessingConfig,
    image_size: int,
) -> np.ndarray:
    """Apply the configured preprocessing pipeline to a grayscale image.

    Returns a float32 array of shape (image_size, image_size). If
    ``cfg.normalize`` is True the output is scaled to [0, 1]; otherwise it is a
    float copy of the 0-255 image.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = resize(img, image_size)

    # Ensure uint8 for the OpenCV ops that require it.
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if cfg.clahe:
        img = apply_clahe(img, cfg.clahe_clip_limit, cfg.clahe_tile_grid)

    img = denoise(img, cfg.denoise, cfg.denoise_ksize)

    out = img.astype(np.float32)
    if cfg.normalize:
        out /= 255.0
    return out


def load_and_preprocess(
    path: str | Path,
    cfg: PreprocessingConfig,
    image_size: int,
) -> np.ndarray:
    """Convenience: load from disk then preprocess."""
    return preprocess_image(load_grayscale(path), cfg, image_size)
