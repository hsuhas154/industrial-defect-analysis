"""Build every classical-branch figure from saved results and artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

from idefect.config import Config
from idefect.data.datasets import build_split, load_split_arrays
from idefect.evaluation.classification import load_predictions
from idefect.features.registry import extract_many, feature_names, parse_feature_set
from idefect.visualization import classical as V


def _final_metrics(results_dir: Path) -> List[Dict]:
    out = []
    for p in sorted(results_dir.glob("*_metrics.json")):
        with open(p) as fh:
            m = json.load(fh)
        out.append(m)
    return out


def roc_pr_figures(results_dir: Path, figures_dir: Path) -> List[Path]:
    """One-vs-rest ROC and PR curves for every model with saved probabilities."""
    made = []
    for m in _final_metrics(results_dir):
        csv = results_dir / f"{m['model']}_test_predictions.csv"
        if not csv.exists():
            continue
        rows = load_predictions(csv)
        classes = m["class_names"]
        if not rows or f"p_{classes[0]}" not in rows[0] or rows[0][f"p_{classes[0]}"] == "":
            continue
        y = np.array([classes.index(r["true"]) for r in rows])
        P = np.array([[float(r[f"p_{c}"]) for c in classes] for r in rows])
        made.append(V.plot_roc_pr(y, P, classes, figures_dir / "roc_pr" / f"{m['model']}_roc_pr.png",
                                  m.get("display_name", m["model"])))
    return made


def make_classical_figures(cfg: Config) -> Dict[str, object]:
    results_dir = Path(cfg.output.results_dir)
    fig_dir = Path(cfg.output.figures_dir) / "classical"
    out: Dict[str, object] = {}

    grid_file = results_dir / "classical" / "grid.json"
    if grid_file.exists():
        rows = json.loads(grid_file.read_text())
        out["heat_cv"] = V.plot_grid_heatmap(rows, "cv_macro_f1_mean", fig_dir / "grid_cv_macro_f1.png",
                                             "Cross-validation macro-F1 (training split): feature set x classifier")
        out["heat_test"] = V.plot_grid_heatmap(rows, "test_macro_f1", fig_dir / "grid_test_macro_f1.png",
                                               "Test macro-F1: feature set x classifier")
    out["roc_pr"] = roc_pr_figures(results_dir, Path(cfg.output.figures_dir))

    _, class_names, split, groups = build_split(cfg)
    arrays = {p: load_split_arrays(getattr(split, p), cfg.preprocessing, cfg.data.image_size)
              for p in ("train", "val", "test")}
    X_all = np.concatenate([arrays[p][0] for p in ("train", "val", "test")])
    y_all = np.concatenate([arrays[p][1] for p in ("train", "val", "test")])
    feats = {d: extract_many(X_all, d, cfg) for d in ("hog", "lbp", "glcm")}
    out["projections"] = V.plot_projections(feats, y_all, class_names, fig_dir / "feature_projections.png",
                                            seed=cfg.seed)

    Xtr, ytr = arrays["train"]
    picks = [int(np.nonzero(ytr == k)[0][0]) for k in range(len(class_names))]
    more = [i for k in range(len(class_names)) for i in np.nonzero(ytr == k)[0][:60]]
    out["lbp"] = V.plot_lbp([Xtr[i] for i in picks + more], [int(ytr[i]) for i in picks + more],
                            class_names, cfg.lbp, fig_dir / "lbp_codes_and_histograms.png")
    out["glcm"] = V.plot_glcm_distributions(extract_many(Xtr, "glcm", cfg),
                                            feature_names("glcm", cfg), ytr, class_names,
                                            fig_dir / "glcm_distributions.png")

    out["depth"] = _depth_curve(cfg, results_dir, split, groups, arrays, fig_dir)
    out["importances"] = _importances(cfg, results_dir, fig_dir)

    abl_dir = results_dir / "classical"
    for key, group_key, title in (("preprocessing", "preprocessing", "Preprocessing ablation"),
                                  ("resolution", "image_size", "Input resolution ablation"),
                                  ("hog", "hog", "HOG parameter study (SVM)")):
        f = abl_dir / f"ablation_{key}.json"
        if f.exists():
            out[f"ablation_{key}"] = V.plot_ablation(json.loads(f.read_text()), group_key,
                                                     fig_dir / f"ablation_{key}.png", title)
    return out


def _depth_curve(cfg, results_dir, split, groups, arrays, fig_dir):
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold, cross_validate
    from sklearn.tree import DecisionTreeClassifier

    from idefect.training.train_classical_ml import group_ids

    finals = [m for m in _final_metrics(results_dir) if m.get("classifier") == "decision_tree"]
    fs = finals[0]["feature_set"] if finals else "hog"
    Xtr = extract_many(arrays["train"][0], fs, cfg)
    ytr = arrays["train"][1]
    g = group_ids(split.train, groups)
    depths = [1, 2, 3, 4, 6, 8, 10, 15, 20, None]
    cv = StratifiedGroupKFold(n_splits=cfg.classical_ml.cv_folds, shuffle=True, random_state=cfg.seed)
    tr, cm, cs = [], [], []
    for d in depths:
        est = DecisionTreeClassifier(max_depth=d, random_state=cfg.seed)
        r = cross_validate(est, Xtr, ytr, groups=g, cv=cv, scoring="f1_macro",
                           n_jobs=cfg.classical_ml.n_jobs)
        cm.append(float(r["test_score"].mean()))
        cs.append(float(r["test_score"].std()))
        tr.append(float(f1_score(ytr, est.fit(Xtr, ytr).predict(Xtr), average="macro")))
    with open(results_dir / "classical" / "decision_tree_depth_curve.json", "w") as fh:
        json.dump({"feature_set": fs, "depths": [str(d) for d in depths], "train": tr,
                   "cv_mean": cm, "cv_std": cs}, fh, indent=2)
    return V.plot_depth_curve(["none" if d is None else d for d in depths], tr, cm, cs,
                              fig_dir / "decision_tree_depth_curve.png",
                              f"Decision tree depth versus overfitting ({fs.upper()} features)")


def _importances(cfg, results_dir, fig_dir):
    made = []
    for m in _final_metrics(results_dir):
        if m.get("classifier") not in ("random_forest", "xgboost"):
            continue
        bundle = joblib.load(results_dir / f"{m['model']}.joblib")
        est = bundle["model"]
        imp = getattr(est, "feature_importances_", None)
        if imp is None:
            continue
        fs = bundle["feature_set"]
        hog_dim = len(imp) - sum(
            len(feature_names(p, cfg)) for p in parse_feature_set(fs) if p != "hog")
        names = feature_names(fs, cfg, hog_dim=hog_dim)
        made.append(V.plot_importances(np.asarray(imp), names,
                                       fig_dir / f"importances_{m['model']}.png",
                                       f"Top features: {m.get('display_name', m['model'])}"))
    return made
