"""Histogram of Oriented Gradients (HOG) feature extraction.

HOG encodes the distribution of local gradient orientations. For industrial
surface defects (cracks, scratches, patch boundaries) this edge/gradient
prior is a strong, illumination-robust descriptor for a classical classifier.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from skimage.feature import hog

from idefect.config import HOGConfig


def extract_hog(img: np.ndarray, cfg: HOGConfig) -> np.ndarray:
    """Extract a 1-D HOG feature vector from a single grayscale image.

    ``img`` is expected to be a 2-D float array (already resized/normalized).
    """
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {img.shape}")
    features = hog(
        img,
        orientations=cfg.orientations,
        pixels_per_cell=tuple(cfg.pixels_per_cell),
        cells_per_block=tuple(cfg.cells_per_block),
        block_norm=cfg.block_norm,
        transform_sqrt=cfg.transform_sqrt,
        feature_vector=True,
    )
    return features.astype(np.float32)


def hog_visualization(img: np.ndarray, cfg: HOGConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(features, hog_image)`` for visualization purposes."""
    features, hog_image = hog(
        img,
        orientations=cfg.orientations,
        pixels_per_cell=tuple(cfg.pixels_per_cell),
        cells_per_block=tuple(cfg.cells_per_block),
        block_norm=cfg.block_norm,
        transform_sqrt=cfg.transform_sqrt,
        visualize=True,
        feature_vector=True,
    )
    return features.astype(np.float32), hog_image


def extract_hog_batch(images: np.ndarray, cfg: HOGConfig) -> np.ndarray:
    """Extract HOG features for a batch of images ``(N, H, W)``.

    Returns an array of shape ``(N, feature_dim)``.
    """
    if images.ndim != 3:
        raise ValueError(
            f"expected a batch of shape (N, H, W), got {images.shape}"
        )
    feats = [extract_hog(images[i], cfg) for i in range(images.shape[0])]
    return np.stack(feats).astype(np.float32)
