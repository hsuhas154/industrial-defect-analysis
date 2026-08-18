"""Train the classical branch: preprocessing -> HOG -> {Decision Tree, SVM}.

Both classifiers are trained on the *same* HOG features from the *same*
deterministic split, so the comparison is fair. Fitted models are saved as
self-contained joblib bundles that also carry the preprocessing/HOG config, so
inference can reproduce the exact feature pipeline.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np

from idefect.classical_ml import build_decision_tree, build_svm
from idefect.config import Config
from idefect.data.datasets import load_samples, load_split_arrays, make_splits
from idefect.evaluation.classification import (
    compute_metrics,
    format_report,
    save_metrics,
    summarize,
)
from idefect.features.hog import extract_hog_batch, hog_visualization
from idefect.seed import set_seed
from idefect.visualization.plots import plot_confusion_matrix, plot_hog


def _save_bundle(model, kind: str, cfg: Config, class_names, path: Path) -> None:
    bundle = {
        "kind": kind,  # "hog+decision_tree" | "hog+svm"
        "model": model,
        "class_names": list(class_names),
        "image_size": cfg.data.image_size,
        "hog": cfg.hog,
        "preprocessing": cfg.preprocessing,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def train_classical_ml(cfg: Config) -> Dict[str, Dict]:
    """Train Decision Tree and SVM baselines. Returns a dict of test metrics."""
    set_seed(cfg.seed)
    results_dir = Path(cfg.output.results_dir)
    figures_dir = Path(cfg.output.figures_dir)

    print(f"[data] scanning {cfg.data.root}")
    samples, class_names = load_samples(cfg.data.root)
    split = make_splits(
        samples,
        class_names,
        cfg.data.train_frac,
        cfg.data.val_frac,
        cfg.data.test_frac,
        cfg.seed,
    )
    print(f"[data] {split.summary()}")

    print("[preprocess] loading + preprocessing images ...")
    X_train, y_train = load_split_arrays(split.train, cfg.preprocessing, cfg.data.image_size)
    X_test, y_test = load_split_arrays(split.test, cfg.preprocessing, cfg.data.image_size)

    print("[features] extracting HOG ...")
    t0 = time.time()
    H_train = extract_hog_batch(X_train, cfg.hog)
    H_test = extract_hog_batch(X_test, cfg.hog)
    print(
        f"[features] HOG dim={H_train.shape[1]} "
        f"({time.time() - t0:.1f}s for {len(H_train) + len(H_test)} images)"
    )

    # Save one HOG visualization for the report.
    _, hog_img = hog_visualization(X_train[0], cfg.hog)
    plot_hog(X_train[0], hog_img, figures_dir / "hog_example.png")

    all_metrics: Dict[str, Dict] = {}

    specs = [
        ("hog+decision_tree", build_decision_tree(cfg.classical_ml.decision_tree, cfg.seed)),
        ("hog+svm", build_svm(cfg.classical_ml.svm, cfg.seed)),
    ]

    for kind, model in specs:
        print(f"\n[train] fitting {kind} ...")
        t0 = time.time()
        model.fit(H_train, y_train)
        fit_time = time.time() - t0

        y_pred = model.predict(H_test)
        metrics = compute_metrics(y_test, y_pred, class_names)
        metrics["fit_time_sec"] = round(fit_time, 3)
        metrics["feature_dim"] = int(H_train.shape[1])
        all_metrics[kind] = metrics

        print(f"[eval:{kind}] {summarize(metrics)}")
        print(format_report(y_test, y_pred, class_names))

        save_metrics(metrics, results_dir / f"{kind}_metrics.json")
        _save_bundle(model, kind, cfg, class_names, results_dir / f"{kind}.joblib")
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            class_names,
            figures_dir / f"{kind}_confusion_matrix.png",
            title=f"{kind} — confusion matrix",
        )

    return all_metrics
