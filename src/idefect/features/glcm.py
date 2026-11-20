"""Grey-level co-occurrence matrix (GLCM) texture statistics.

A GLCM counts how often grey level i occurs next to grey level j at a given
distance d and angle theta. Summary statistics of the (normalized) matrix
describe texture:

* contrast      sum P(i,j) (i-j)^2        large for strong local variation
* dissimilarity sum P(i,j) |i-j|          like contrast, linear in the difference
* homogeneity   sum P(i,j) / (1+(i-j)^2)  large when neighbours are similar
* ASM           sum P(i,j)^2              angular second moment: uniformity
* energy        sqrt(ASM)
* correlation   linear dependency of neighbouring grey levels

Each statistic is computed for every distance and the four angles 0, 45, 90
and 135 degrees. Per statistic and distance the mean over angles (a
rotation-invariant value) and the range over angles (how directional the
texture is, for example scratches and rolled-in streaks) are kept. With six
statistics and four distances that gives 48 features. Images are quantized to
``levels`` grey levels first, which keeps the matrices well populated.
"""

from __future__ import annotations

from typing import List

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from idefect.config import GLCMConfig

ANGLES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def _quantize(img: np.ndarray, levels: int) -> np.ndarray:
    x = img.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    return np.clip((x * levels).astype(np.int32), 0, levels - 1).astype(np.uint8)


def glcm_matrix(img: np.ndarray, cfg: GLCMConfig) -> np.ndarray:
    """Normalized, symmetric GLCMs of shape (levels, levels, n_distances, 4)."""
    q = _quantize(img, cfg.levels)
    return graycomatrix(q, distances=list(cfg.distances), angles=ANGLES, levels=cfg.levels,
                        symmetric=True, normed=True)


def extract_glcm(img: np.ndarray, cfg: GLCMConfig) -> np.ndarray:
    P = glcm_matrix(img, cfg)
    feats = []
    for prop in cfg.properties:
        v = graycoprops(P, prop)  # (n_distances, 4)
        feats.append(v.mean(axis=1))
        feats.append(v.max(axis=1) - v.min(axis=1))
    return np.concatenate(feats).astype(np.float32)


def glcm_feature_names(cfg: GLCMConfig) -> List[str]:
    names = []
    for prop in cfg.properties:
        names += [f"glcm_{prop}_d{d}_mean" for d in cfg.distances]
        names += [f"glcm_{prop}_d{d}_range" for d in cfg.distances]
    return names
