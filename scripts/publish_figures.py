#!/usr/bin/env python3
"""Copy the key result figures into docs/figures for the README.

Figures are copied byte for byte at full resolution.

Usage:
    python scripts/publish_figures.py --config configs/neu_baseline.yaml
"""

import argparse
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401

from idefect.config import load_config

KEY_FIGURES = [
    "model_comparison.png",
    "eda/samples_by_class.png",
    "eda/preprocessing_comparison.png",
    "duplicates/group_sizes.png",
    "duplicates/same_pixel_identical.png",
    "duplicates/same_grain_weakest.png",
    "duplicates/distinct_closest.png",
    "classical/grid_cv_macro_f1.png",
    "classical/grid_test_macro_f1.png",
    "classical/feature_projections.png",
    "classical/lbp_codes_and_histograms.png",
    "classical/glcm_distributions.png",
    "classical/decision_tree_depth_curve.png",
    "classical/ablation_preprocessing.png",
    "classical/ablation_resolution.png",
    "classical/ablation_hog.png",
    "classical/importances_*.png",
    "roc_pr/*.png",
    "custom_cnn_confusion_matrix.png",
    "custom_cnn_training_curves.png",
    "*_confusion_matrix.png",
    "errors/*_misclassified.png",
]


def publish(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst.stat().st_size


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/neu_baseline.yaml")
    p.add_argument("--out", default="docs/figures")
    args = p.parse_args()
    fig_dir = Path(load_config(args.config).output.figures_dir)
    seen = set()
    for rel in KEY_FIGURES:
        matches = sorted(fig_dir.glob(rel))
        if not matches:
            print(f"[skip] {fig_dir / rel} (not generated yet)")
        for src in matches:
            if src in seen:
                continue
            seen.add(src)
            size = publish(src, Path(args.out) / src.name)
            print(f"[ok] {Path(args.out) / src.name} ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
