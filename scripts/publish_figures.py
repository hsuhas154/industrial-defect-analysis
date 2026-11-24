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
    # classical models (Phase 2) and the CNNs the write-up discusses
    "lbp+glcm+*_confusion_matrix.png",
    "hog+lbp+glcm+*_confusion_matrix.png",
    "errors/lbp+glcm+*_misclassified.png",
    "errors/hog+lbp+glcm+*_misclassified.png",
    "errors/custom_cnn_misclassified.png",
    *[f"{m}_{kind}.png" for m in ("vgg", "resnet18_imagenet_full", "densenet121_imagenet_full")
      for kind in ("confusion_matrix", "training_curves")],
    "errors/vgg_misclassified.png",
    "errors/resnet18_imagenet_full_misclassified.png",
    # Phase 3: CNN study
    "cnn/accuracy_vs_cpu_latency.png",
    "cnn/accuracy_vs_params.png",
    "cnn/data_curve.png",
    "cnn/cnn_resolution.png",
    "cnn/cnn_clahe.png",
    "cnn/gradcam_comparison.png",
    "cnn/gradcam_vgg.png",
    "cnn/gradcam_densenet121_imagenet_full.png",
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
