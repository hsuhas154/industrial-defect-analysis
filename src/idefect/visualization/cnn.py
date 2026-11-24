"""CNN-branch figures: Grad-CAM grids and cross-model comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from idefect.visualization.plots import plt


def _save(fig, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _overlay(ax, img: np.ndarray, cam: np.ndarray, title: str = "", fontsize: int = 7) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=fontsize)
    ax.axis("off")


def plot_gradcam_grid(images: np.ndarray, cams: np.ndarray, titles: Sequence[str], out_path,
                      title: str, cols: int = 4) -> Path:
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.75 * rows), squeeze=False,
                             layout="constrained")
    for i, ax in enumerate(axes.flat):
        if i < n:
            _overlay(ax, images[i], cams[i], titles[i])
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=10)
    return _save(fig, out_path)


def plot_gradcam_comparison(images: Sequence[np.ndarray], names: Sequence[str],
                            per_model: Dict[str, tuple], out_path) -> Path:
    """Rows: images; first column the input, then one Grad-CAM per model with
    its predicted class."""
    models = list(per_model)
    rows, cols = len(images), len(models) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(1.9 * cols, 2.0 * rows), squeeze=False,
                             layout="constrained")
    for r, (img, name) in enumerate(zip(images, names)):
        ax = axes[r, 0]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=7)
        ax.axis("off")
        for c, m in enumerate(models, start=1):
            cams, preds = per_model[m]
            _overlay(axes[r, c], img, cams[r], (m + "\n" if r == 0 else "") + f"pred {preds[r]}", fontsize=6)
    fig.suptitle("Grad-CAM across CNNs (test images, one per class, and Sc_44)", fontsize=10)
    return _save(fig, out_path)


def plot_accuracy_vs_cost(rows: List[Dict], x_key: str, x_label: str, out_path, title: str) -> Path:
    """Test macro-F1 (seed mean where available) against a cost, log x-axis."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    for r in rows:
        if r.get(x_key) is None or r.get("test_macro_f1") is None:
            continue
        y = r.get("test_macro_f1_mean") or r["test_macro_f1"]
        err = r.get("test_macro_f1_std") or 0
        color = "tab:blue" if r["branch"] == "cnn" else "tab:orange"
        ax.errorbar(r[x_key], y, yerr=err, fmt="o", color=color, capsize=2, ms=5)
        ax.annotate(r.get("display_name", r["model"]), (r[x_key], y), fontsize=6.5,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set(xlabel=x_label, ylabel="test macro-F1 (mean over seeds for CNNs)", title=title)
    ax.grid(alpha=0.3, which="both")
    from matplotlib.lines import Line2D

    ax.legend(handles=[Line2D([], [], color="tab:blue", marker="o", ls="", label="CNN"),
                       Line2D([], [], color="tab:orange", marker="o", ls="", label="classical")],
              loc="lower right", fontsize=8)
    return _save(fig, out_path)
