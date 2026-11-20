"""Local Binary Patterns (LBP) texture descriptor.

For every pixel, the P neighbours on a circle of radius R are thresholded
against the centre value, giving a P-bit pattern. "Uniform" patterns (at most
two 0/1 transitions around the circle) correspond to edges, corners, spots and
flat areas; with the rotation-invariant uniform mapping ("riu2") each pattern
is reduced to its number of set bits, so there are P + 2 codes (P + 1 uniform
codes and one bin for all non-uniform patterns). The descriptor is the
normalized histogram of codes, optionally per cell of a spatial grid, computed
at several (P, R) scales and concatenated.

Why LBP for steel surfaces? It encodes micro-texture (pits, grain, fine
cracks) and is invariant to monotonic grey-level changes, so illumination
differences between images matter little. Rotation invariance suits defects
that have no preferred orientation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from skimage.feature import local_binary_pattern

from idefect.config import LBPConfig


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    if x.max() <= 1.0:
        x = x * 255.0
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def lbp_codes(img: np.ndarray, points: int, radius: float, method: str = "uniform") -> np.ndarray:
    """LBP code image for one (P, R) scale."""
    return local_binary_pattern(_to_uint8(img), points, radius, method=method)


def n_bins(points: int, method: str) -> int:
    if method == "uniform":
        return points + 2
    if method == "nri_uniform":
        return points * (points - 1) + 3
    return 2 ** points


def extract_lbp(img: np.ndarray, cfg: LBPConfig) -> np.ndarray:
    """Concatenated normalized LBP histograms over all scales and grid cells."""
    feats: List[np.ndarray] = []
    g = max(1, int(cfg.grid))
    for points, radius in cfg.points_radii:
        codes = lbp_codes(img, int(points), float(radius), cfg.method)
        bins = n_bins(int(points), cfg.method)
        H, W = codes.shape
        for gy in range(g):
            for gx in range(g):
                cell = codes[gy * H // g:(gy + 1) * H // g, gx * W // g:(gx + 1) * W // g]
                hist = np.bincount(cell.astype(np.int64).ravel(), minlength=bins)[:bins]
                feats.append(hist / max(1, hist.sum()))
    return np.concatenate(feats).astype(np.float32)


def lbp_feature_names(cfg: LBPConfig) -> List[str]:
    names = []
    g = max(1, int(cfg.grid))
    for points, radius in cfg.points_radii:
        for gy in range(g):
            for gx in range(g):
                cell = f"_c{gy}{gx}" if g > 1 else ""
                for b in range(n_bins(int(points), cfg.method)):
                    names.append(f"lbp_P{points}R{radius:g}{cell}_bin{b}")
    return names


def lbp_dimension(cfg: LBPConfig) -> int:
    g = max(1, int(cfg.grid))
    return sum(n_bins(int(p), cfg.method) for p, _ in cfg.points_radii) * g * g


def lbp_scales(cfg: LBPConfig) -> List[Tuple[int, float]]:
    return [(int(p), float(r)) for p, r in cfg.points_radii]
