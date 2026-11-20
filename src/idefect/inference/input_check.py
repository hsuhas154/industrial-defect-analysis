"""Sanity checks for images given to a trained model at inference time.

All models are *closed-set*: they always answer with one of the six NEU
classes, even for a photo of something else. These checks flag inputs that do
not look like the training data, so the app can say "this prediction may be
meaningless" instead of showing a confident label:

* colour content (NEU images are grayscale);
* global brightness and contrast outside the range seen in the training
  images, or far from their median (robust z-score). NEU brightness varies
  so widely (mean grey level 34 to 246) that the z-score alone can never
  fire for an 8-bit image, so the observed range is checked as well;
* confidence below what the model shows on 95 percent of its *correct*
  validation predictions;
* for decision trees and k-NN: probabilities are leaf class fractions or
  neighbour votes, not calibrated confidences (with k = 1 they are always
  0 or 1).

The reference statistics are computed on training images and validation
predictions only and are stored inside the model artifact.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np


def image_stats(gray: np.ndarray) -> Dict[str, float]:
    x = gray.astype(np.float32)
    return {"mean": float(x.mean()), "std": float(x.std())}


def reference_stats(paths: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Median and robust spread (MAD scaled to a standard deviation) of
    brightness and contrast over a set of training images."""
    vals = {"mean": [], "std": []}
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        s = image_stats(img)
        vals["mean"].append(s["mean"])
        vals["std"].append(s["std"])
    out = {}
    for k, v in vals.items():
        v = np.asarray(v)
        med = float(np.median(v))
        out[k] = {"median": med, "mad_sigma": float(1.4826 * np.median(np.abs(v - med)) + 1e-6),
                  "min": float(v.min()), "max": float(v.max())}
    return out


def confidence_floor(proba: np.ndarray, y_true: np.ndarray, q: float = 0.05) -> Optional[float]:
    """The q-quantile of the predicted-class probability over correct validation predictions."""
    if proba is None or len(proba) == 0:
        return None
    pred = proba.argmax(1)
    ok = pred == np.asarray(y_true)
    if not ok.any():
        return None
    return float(np.quantile(proba[ok, :].max(1), q))


COLOUR_THRESHOLD = 8.0  # mean chroma above which an input counts as a colour image


def is_colour(image: np.ndarray) -> bool:
    return image.ndim == 3 and colourfulness(image) > COLOUR_THRESHOLD


def colourfulness(rgb: np.ndarray) -> float:
    """Mean chroma (0 for perfectly grey images)."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return 0.0
    hsv = cv2.cvtColor(rgb[..., :3].astype(np.uint8), cv2.COLOR_RGB2HSV)
    return float((hsv[..., 1].astype(np.float32) * hsv[..., 2] / 255.0).mean())


def check_input(image: np.ndarray, reference: Optional[Dict], confidence: Optional[float],
                model_kind: str = "", z_limit: float = 4.0,
                n_neighbors: Optional[int] = None) -> List[str]:
    """Human-readable warnings for one input (empty list: nothing unusual)."""
    warnings: List[str] = []
    if is_colour(image):
        warnings.append("The image is in colour; NEU training images are grayscale steel "
                        "close-ups, so the prediction may not be meaningful.")
    gray = image if image.ndim == 2 else cv2.cvtColor(image[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if reference and "image" in reference:
        stats = image_stats(gray)
        for key, label in (("mean", "brightness"), ("std", "contrast")):
            ref = reference["image"][key]
            value = stats[key]
            z = (value - ref["median"]) / ref["mad_sigma"]
            lo, hi = ref.get("min"), ref.get("max")
            outside = lo is not None and hi is not None and not (lo <= value <= hi)
            if outside:
                warnings.append(f"Unusual {label} for this model ({value:.0f}; training images "
                                f"range from {lo:.0f} to {hi:.0f}).")
            elif abs(z) > z_limit:
                warnings.append(f"Unusual {label} for this model ({value:.0f} versus "
                                f"{ref['median']:.0f} typical in training).")
    floor = (reference or {}).get("confidence_floor")
    if confidence is not None and floor is not None and confidence < floor:
        warnings.append(f"Low confidence: below what the model shows on 95% of its correct "
                        f"validation predictions ({floor * 100:.0f}%). Treat the label with caution.")
    if "decision_tree" in model_kind:
        warnings.append("Decision tree probabilities are the class mix of a single leaf "
                        "(mostly 0% or 100%), not calibrated confidences.")
    if "knn" in model_kind:
        if n_neighbors == 1:
            detail = "this model uses k = 1, so they are always 0% or 100%"
        elif n_neighbors:
            detail = f"this model uses k = {n_neighbors}"
        else:
            detail = "their resolution is 1/k"
        warnings.append(f"k-NN probabilities are the share of the nearest training images in each "
                        f"class ({detail}), not calibrated confidences.")
    return warnings
