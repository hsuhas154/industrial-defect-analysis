#!/usr/bin/env python3
"""Copy the key result figures into docs/figures for the README.

Figures are downscaled and, when they contain no colour, stored as
single-channel PNGs so the committed copies stay small.

Usage:
    python scripts/publish_figures.py --config configs/neu_baseline.yaml
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401

from idefect.config import load_config

KEY_FIGURES = [
    "model_comparison.png",
    "eda/samples_by_class.png",
    "eda/preprocessing_comparison.png",
    "custom_cnn_confusion_matrix.png",
    "custom_cnn_training_curves.png",
    "hog+svm_confusion_matrix.png",
    "errors/hog+svm_misclassified.png",
    "errors/custom_cnn_misclassified.png",
    "hog_example.png",
]


def publish(src: Path, dst: Path, max_width: int = 1400) -> int:
    im = Image.open(src).convert("RGB")
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
    arr = np.asarray(im)
    if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
        im = im.convert("L")
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, optimize=True)
    return dst.stat().st_size


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/neu_baseline.yaml")
    p.add_argument("--out", default="docs/figures")
    args = p.parse_args()
    fig_dir = Path(load_config(args.config).output.figures_dir)
    for rel in KEY_FIGURES:
        src = fig_dir / rel
        if not src.exists():
            print(f"[skip] {src} (not generated yet)")
            continue
        size = publish(src, Path(args.out) / Path(rel).name)
        print(f"[ok] {Path(args.out) / Path(rel).name} ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
