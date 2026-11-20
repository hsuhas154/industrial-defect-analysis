"""Exploratory data analysis figures.

* class balance bar chart;
* sample grid by class;
* preprocessing and classical-CV comparison (raw, CLAHE, filters, edges,
  morphology) for one image per class;
* per-class intensity histograms;
* per-class mean images.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from idefect.data import cv_ops
from idefect.data.datasets import Sample
from idefect.data.preprocessing import apply_clahe, denoise, load_grayscale, resize
from idefect.visualization.plots import _ensure_parent, plt, tidy


def _by_class(samples: Sequence[Sample], n_classes: int) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {i: [] for i in range(n_classes)}
    for p, y in samples:
        out[y].append(p)
    for i in out:
        out[i].sort()
    return out


def plot_class_balance(counts: Dict[str, int], out_path: str | Path) -> Path:
    """Bar chart of image counts per class."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = list(counts)
    ax.bar(names, [counts[n] for n in names], color="#4C72B0")
    for i, n in enumerate(names):
        ax.text(i, counts[n], str(counts[n]), ha="center", va="bottom", fontsize=8)
    ax.set(ylabel="images", title="Images per class")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    tidy(fig)
    out_path = _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_samples_by_class(
    samples: Sequence[Sample],
    class_names: Sequence[str],
    out_path: str | Path,
    per_class: int = 6,
    seed: int = 0,
) -> Path:
    """Grid with one row per class and ``per_class`` random examples per row."""
    groups = _by_class(samples, len(class_names))
    rng = random.Random(seed)
    rows = len(class_names)
    fig, axes = plt.subplots(rows, per_class, figsize=(1.7 * per_class, 1.8 * rows))
    axes = np.atleast_2d(axes)
    for r in range(rows):
        picks = rng.sample(groups[r], min(per_class, len(groups[r])))
        for c in range(per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(picks):
                ax.imshow(load_grayscale(picks[c]), cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_title(class_names[r], fontsize=9, loc="left")
    fig.suptitle("Sample images by class")
    tidy(fig)
    out_path = _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def preprocessing_variants(img: np.ndarray, image_size: int) -> Dict[str, np.ndarray]:
    """The classical-CV views shown in the comparison figure, in display order."""
    base = resize(img, image_size)
    return {
        "raw (resized)": base,
        "CLAHE": apply_clahe(base, 2.0, 8),
        "Gaussian 5x5": denoise(base, "gaussian", 5),
        "median 5x5": denoise(base, "median", 5),
        "Sobel magnitude": cv_ops.sobel_magnitude(base),
        "Canny": cv_ops.canny_edges(base),
        "top-hat (stretched)": _stretch(cv_ops.morphology(base, "tophat", 9)),
        "black-hat (stretched)": _stretch(cv_ops.morphology(base, "blackhat", 9)),
    }


def _stretch(img: np.ndarray) -> np.ndarray:
    """Min-max contrast stretch for display; morphology residues are faint."""
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return img
    return ((img.astype(np.float32) - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def plot_preprocessing_comparison(
    samples: Sequence[Sample],
    class_names: Sequence[str],
    out_path: str | Path,
    image_size: int = 128,
    seed: int = 0,
) -> Path:
    """One row per class, one column per preprocessing / CV operation."""
    groups = _by_class(samples, len(class_names))
    rng = random.Random(seed)
    rows = len(class_names)
    variants0 = preprocessing_variants(load_grayscale(groups[0][0]), image_size)
    cols = len(variants0)
    fig, axes = plt.subplots(rows, cols, figsize=(1.75 * cols, 1.85 * rows))
    axes = np.atleast_2d(axes)
    for r in range(rows):
        img = load_grayscale(rng.choice(groups[r]))
        for c, (name, view) in enumerate(preprocessing_variants(img, image_size).items()):
            ax = axes[r, c]
            ax.imshow(view, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=8)
            if c == 0:
                ax.set_ylabel(class_names[r], fontsize=8)
    fig.suptitle("Preprocessing and classical CV operations")
    tidy(fig)
    out_path = _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_intensity_histograms(
    samples: Sequence[Sample],
    class_names: Sequence[str],
    out_path: str | Path,
    max_per_class: int = 300,
) -> Path:
    """Mean normalized intensity histogram per class (64 bins)."""
    groups = _by_class(samples, len(class_names))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    centers = np.arange(64) * 4 + 2
    for i, name in enumerate(class_names):
        hists = [
            cv_ops.intensity_histogram(load_grayscale(p), bins=64)
            for p in groups[i][:max_per_class]
        ]
        ax.plot(centers, np.mean(hists, axis=0), label=name, linewidth=1.4)
    ax.set(xlabel="intensity", ylabel="fraction of pixels",
           title="Mean intensity histogram per class")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    tidy(fig)
    out_path = _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_class_mean_images(
    samples: Sequence[Sample],
    class_names: Sequence[str],
    out_path: str | Path,
    image_size: int = 128,
) -> Path:
    """Pixel-wise mean image per class. Structured defects such as scratches
    average out, which shows how much each class depends on location."""
    groups = _by_class(samples, len(class_names))
    fig, axes = plt.subplots(1, len(class_names), figsize=(2.0 * len(class_names), 2.4))
    axes = np.atleast_1d(axes)
    for i, name in enumerate(class_names):
        stack = [resize(load_grayscale(p), image_size).astype(np.float32) for p in groups[i]]
        axes[i].imshow(np.mean(stack, axis=0), cmap="gray")
        axes[i].set_title(name, fontsize=8)
        axes[i].axis("off")
    fig.suptitle("Mean image per class (each panel contrast-stretched)")
    tidy(fig)
    out_path = _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
