"""Classical computer-vision operations: edges, morphology, histograms.

These are the building blocks the spec lists under Stage A beyond basic
preprocessing. They are used for exploratory figures now and as optional
"edge-enhanced" inputs for the classical branch later.

All functions take and return single-channel ``uint8`` images unless noted.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

_MORPH_OPS = {
    "erode": cv2.MORPH_ERODE,
    "dilate": cv2.MORPH_DILATE,
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "gradient": cv2.MORPH_GRADIENT,
    "tophat": cv2.MORPH_TOPHAT,
    "blackhat": cv2.MORPH_BLACKHAT,
}

_KERNEL_SHAPES = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a float image in [0, 1] or [0, 255] (or any uint8) to uint8."""
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def canny_edges(img: np.ndarray, low: int | None = None, high: int | None = None) -> np.ndarray:
    """Canny edge map. If thresholds are omitted they are set from the median
    intensity (the common "auto Canny" heuristic: 0.67x and 1.33x median)."""
    img = to_uint8(img)
    if low is None or high is None:
        med = float(np.median(img))
        low = int(max(0, 0.67 * med))
        high = int(min(255, 1.33 * med))
    return cv2.Canny(img, low, high)


def sobel_magnitude(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Gradient magnitude from horizontal and vertical Sobel filters, scaled to uint8."""
    img = to_uint8(img).astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def laplacian(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Absolute Laplacian response (second-derivative edges), scaled to uint8."""
    lap = cv2.Laplacian(to_uint8(img), cv2.CV_32F, ksize=ksize)
    return cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def morphology(img: np.ndarray, op: str, ksize: int = 5, shape: str = "ellipse") -> np.ndarray:
    """Apply a morphological operation.

    ``op`` is one of erode, dilate, open, close, gradient, tophat, blackhat.
    Top-hat highlights small bright details (for example inclusions) and
    black-hat highlights small dark details (for example pits and scratches).
    """
    if op not in _MORPH_OPS:
        raise ValueError(f"unknown morphology op '{op}'; choose from {sorted(_MORPH_OPS)}")
    if shape not in _KERNEL_SHAPES:
        raise ValueError(f"unknown kernel shape '{shape}'")
    kernel = cv2.getStructuringElement(_KERNEL_SHAPES[shape], (ksize, ksize))
    return cv2.morphologyEx(to_uint8(img), _MORPH_OPS[op], kernel)


def otsu_threshold(img: np.ndarray) -> Tuple[float, np.ndarray]:
    """Global Otsu threshold. Returns ``(threshold, binary_mask)``."""
    t, mask = cv2.threshold(to_uint8(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t), mask


def intensity_histogram(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """Normalized intensity histogram (sums to 1) of a uint8 image."""
    hist = cv2.calcHist([to_uint8(img)], [0], None, [bins], [0, 256]).ravel()
    total = hist.sum()
    return hist / total if total else hist


def intensity_stats(img: np.ndarray) -> Dict[str, float]:
    """Mean, standard deviation, min, max and RMS contrast of a uint8 image."""
    x = to_uint8(img).astype(np.float32)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "rms_contrast": float(x.std() / 255.0),
    }
