"""Matplotlib visualizations written to disk.

Uses the non-interactive Agg backend so plots render on headless machines and
inside WSL without a display server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    class_names: Sequence[str],
    out_path: str | Path,
    title: str = "Confusion matrix",
    normalize: bool = False,
) -> Path:
    """Render a confusion matrix heatmap."""
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text = format(value, fmt) if normalize else format(int(value), fmt)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    out_path = _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_training_curves(history: Dict[str, List[float]], out_path: str | Path) -> Path:
    """Plot train/val loss and accuracy over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path = _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_hog(image: np.ndarray, hog_image: np.ndarray, out_path: str | Path) -> Path:
    """Side-by-side of an image and its HOG visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Preprocessed image")
    ax1.axis("off")
    ax2.imshow(hog_image, cmap="gray")
    ax2.set_title("HOG features")
    ax2.axis("off")
    fig.tight_layout()
    out_path = _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_sample_grid(
    images: Sequence[np.ndarray],
    labels: Sequence[str],
    out_path: str | Path,
    cols: int = 6,
    title: str = "Sample images by class",
) -> Path:
    """Grid of sample images with class captions."""
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img, cmap="gray")
        ax.set_title(label, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    out_path = _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
