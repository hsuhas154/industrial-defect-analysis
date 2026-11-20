"""Train and compare the classical branch: handcrafted features x classifiers.

Protocol (leakage-free by construction):

1. One shared split from :func:`idefect.data.datasets.build_split`, with
   near-duplicate groups kept inside a single split.
2. Images are preprocessed and features extracted per image. Extractors have
   no fitted state, so doing this up front cannot leak anything.
3. For every (feature set, classifier) pair, hyperparameters are chosen by
   grid search with stratified *group* k-fold cross-validation on the training
   split only (duplicate groups never straddle a fold). Scalers live inside
   the pipeline, so they are refitted on each fold's training part.
4. The selected configuration is refitted on the whole training split. The
   validation split gives a second, independent check; the test split is
   evaluated once and never used for any choice.
5. For each classifier the feature set with the best mean cross-validation
   macro-F1 becomes its final model (artifact, metrics, test predictions).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from idefect.classical_ml.models import (
    MODEL_NAMES,
    build_model,
    display_name,
    grid_size,
    with_probabilities,
)
from idefect.config import Config
from idefect.data.datasets import Sample, build_split, load_split_arrays
from idefect.evaluation.classification import (
    compute_metrics,
    format_report,
    save_metrics,
    save_predictions,
    summarize,
)
from idefect.experiment_log import config_hash, environment_info, log_run
from idefect.features.registry import extract_many, extract_one
from idefect.inference.input_check import confidence_floor, reference_stats
from idefect.seed import set_seed
from idefect.visualization.plots import plot_confusion_matrix


def group_ids(samples: Sequence[Sample], groups: Sequence[Sequence[str]]) -> np.ndarray:
    """Integer group id per sample: shared for members of a duplicate group,
    unique otherwise. Used by group-aware cross-validation."""
    gid = {}
    for k, g in enumerate(groups):
        for p in g:
            gid[str(p)] = k
    out, nxt = [], len(groups)
    for p, _ in samples:
        if str(p) in gid:
            out.append(gid[str(p)])
        else:
            out.append(nxt)
            nxt += 1
    return np.asarray(out)


def tune(model: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: Config,
         verbose: bool = True) -> Dict:
    """Grid search with stratified group k-fold CV on (X, y). Returns the
    refitted best estimator plus CV statistics."""
    est, grid, parallel_ok = build_model(model, X.shape[1], cfg.seed, cfg.classical_ml.xgb_device)
    cv = StratifiedGroupKFold(n_splits=cfg.classical_ml.cv_folds, shuffle=True,
                              random_state=cfg.seed)
    search = GridSearchCV(est, grid, scoring="f1_macro", cv=cv, refit=False,
                          n_jobs=cfg.classical_ml.n_jobs if parallel_ok else 1,
                          error_score="raise")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y, groups=groups)
    search_time = time.time() - t0
    i = int(search.best_index_)
    best = with_probabilities(clone(est).set_params(**search.best_params_))
    t1 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best.fit(X, y)
    fit_time = time.time() - t1
    best_params = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    if verbose:
        print(f"    {model:14s} CV macro-F1 {search.cv_results_['mean_test_score'][i]:.4f} "
              f"+- {search.cv_results_['std_test_score'][i]:.4f}  ({grid_size(grid)} configs, "
              f"search {search_time:.1f}s)  best {best_params}")
    return {
        "estimator": best,
        "best_params": best_params,
        "cv_macro_f1_mean": float(search.cv_results_["mean_test_score"][i]),
        "cv_macro_f1_std": float(search.cv_results_["std_test_score"][i]),
        "n_configs": grid_size(grid),
        "search_time_sec": round(search_time, 2),
        "fit_time_sec": round(fit_time, 3),
        "cv_results": {
            "params": [{k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()}
                       for p in search.cv_results_["params"]],
            "mean": search.cv_results_["mean_test_score"].tolist(),
            "std": search.cv_results_["std_test_score"].tolist(),
        },
    }


def _predict_with_proba(est, X):
    y_pred = est.predict(X)
    proba = est.predict_proba(X) if hasattr(est, "predict_proba") else None
    return y_pred, proba


def _latency_ms(est, images: np.ndarray, feature_set: str, cfg: Config, n: int = 100) -> float:
    """Single-image latency (feature extraction + prediction) in milliseconds,
    on preprocessed images, measured sequentially as an app would see it."""
    idx = np.arange(min(n, len(images)))
    t0 = time.perf_counter()
    for i in idx:
        f = extract_one(images[i], feature_set, cfg)[None]
        est.predict(f)
    return 1000.0 * (time.perf_counter() - t0) / max(1, len(idx))


def save_bundle(est, feature_set: str, model: str, cfg: Config, class_names, path: Path,
                reference: Optional[Dict] = None) -> None:
    bundle = {
        "kind": f"{feature_set}+{model}",
        "feature_set": feature_set,
        "model_name": model,
        "model": est,
        "class_names": list(class_names),
        "image_size": cfg.data.image_size,
        "hog": cfg.hog,
        "lbp": cfg.lbp,
        "glcm": cfg.glcm,
        "preprocessing": cfg.preprocessing,
        "config_hash": config_hash(cfg),
        "reference": reference or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def prepare_data(cfg: Config, verbose: bool = True):
    """Shared split, group ids and preprocessed image stacks."""
    _, class_names, split, groups = build_split(cfg)
    if verbose:
        print(f"[data] {split.summary()}; {len(groups)} near-duplicate group(s) kept within one split")
    arrays = {}
    for part in ("train", "val", "test"):
        arrays[part] = load_split_arrays(getattr(split, part), cfg.preprocessing, cfg.data.image_size)
    g_train = group_ids(split.train, groups)
    return class_names, split, groups, arrays, g_train


def train_classical_ml(cfg: Config, config_path: str = "",
                       feature_sets: Optional[List[str]] = None,
                       models: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Run the full (feature set x classifier) comparison. Returns final metrics per model."""
    set_seed(cfg.seed)
    feature_sets = feature_sets or list(cfg.classical_ml.feature_sets)
    models = models or list(cfg.classical_ml.models)
    results_dir = Path(cfg.output.results_dir)
    figures_dir = Path(cfg.output.figures_dir)
    grid_dir = results_dir / "classical"
    grid_dir.mkdir(parents=True, exist_ok=True)

    class_names, split, groups, arrays, g_train = prepare_data(cfg)
    (Xtr_img, y_train), (Xva_img, y_val), (Xte_img, y_test) = arrays["train"], arrays["val"], arrays["test"]

    rows: List[Dict] = []
    fitted: Dict[tuple, Dict] = {}
    for fs in feature_sets:
        t0 = time.time()
        F = {p: extract_many(a[0], fs, cfg, n_jobs=cfg.classical_ml.n_jobs)
             for p, a in arrays.items()}
        print(f"[features] {fs}: dim={F['train'].shape[1]} ({time.time() - t0:.1f}s)")
        for model in models:
            res = tune(model, F["train"], y_train, g_train, cfg)
            est = res["estimator"]
            yv, _ = _predict_with_proba(est, F["val"])
            yt, pt = _predict_with_proba(est, F["test"])
            m_test = compute_metrics(y_test, yt, class_names, y_proba=pt)
            row = {
                "feature_set": fs, "model": model,
                "feature_dim": int(F["train"].shape[1]),
                "cv_macro_f1_mean": res["cv_macro_f1_mean"],
                "cv_macro_f1_std": res["cv_macro_f1_std"],
                "val_macro_f1": float(f1_score(y_val, yv, average="macro")),
                "test_accuracy": m_test["accuracy"],
                "test_macro_f1": m_test["macro_f1"],
                "test_roc_auc": m_test["roc_auc_ovr_macro"],
                "search_time_sec": res["search_time_sec"],
                "fit_time_sec": res["fit_time_sec"],
                "n_configs": res["n_configs"],
                "best_params": res["best_params"],
            }
            rows.append(row)
            fitted[(fs, model)] = {"res": res, "F": F, "y_test_pred": yt, "proba": pt,
                                   "metrics": m_test}
            with open(grid_dir / f"{fs}+{model}_search.json", "w") as fh:
                json.dump({**row, "cv_results": res["cv_results"]}, fh, indent=2)

    with open(grid_dir / "grid.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    _write_grid_csv(rows, grid_dir / "grid.csv")

    # Final model per classifier: best feature set by cross-validation only.
    train_ref = reference_stats([p for p, _ in split.train])
    finals: Dict[str, Dict] = {}
    for model in models:
        cand = [r for r in rows if r["model"] == model]
        best = max(cand, key=lambda r: (r["cv_macro_f1_mean"], -r["feature_dim"]))
        fs = best["feature_set"]
        f = fitted[(fs, model)]
        kind = f"{fs}+{model}"
        metrics = dict(f["metrics"])
        fname, mname = display_name(fs, model)
        metrics.update(
            model=kind,
            display_name=f"{fname} + {mname}",
            branch="classical",
            feature_set=fs,
            classifier=model,
            feature_dim=best["feature_dim"],
            cv_macro_f1_mean=best["cv_macro_f1_mean"],
            cv_macro_f1_std=best["cv_macro_f1_std"],
            val_macro_f1=best["val_macro_f1"],
            best_params=best["best_params"],
            fit_time_sec=best["fit_time_sec"],
            search_time_sec=best["search_time_sec"],
            latency_ms_per_image=round(_latency_ms(f["res"]["estimator"], Xte_img, fs, cfg), 3),
            selection="feature set chosen by mean CV macro-F1 on the training split",
            config_hash=config_hash(cfg),
            environment=environment_info(),
        )
        finals[model] = metrics
        print(f"[final:{kind}] {summarize(metrics)}")
        print(format_report(y_test, f["y_test_pred"], class_names))
        save_metrics(metrics, results_dir / f"{kind}_metrics.json")
        save_predictions(results_dir / f"{kind}_test_predictions.csv",
                         [p for p, _ in split.test], y_test, f["y_test_pred"], class_names, f["proba"])
        pv = f["res"]["estimator"].predict_proba(f["F"]["val"]) \
            if hasattr(f["res"]["estimator"], "predict_proba") else None
        reference = {"image": train_ref,
                     "confidence_floor": confidence_floor(pv, y_val) if pv is not None else None}
        save_bundle(f["res"]["estimator"], fs, model, cfg, class_names,
                    results_dir / f"{kind}.joblib", reference)
        plot_confusion_matrix(metrics["confusion_matrix"], class_names,
                              figures_dir / f"{kind}_confusion_matrix.png",
                              title=f"{fname} + {mname} confusion matrix")
        log_run(cfg.output.logs_dir, kind, cfg, metrics, config_path=config_path)
    with open(grid_dir / "finals.json", "w") as fh:
        json.dump({m: {k: v for k, v in d.items() if k not in ("confusion_matrix", "per_class")}
                   for m, d in finals.items()}, fh, indent=2)
    return finals


def _write_grid_csv(rows: List[Dict], path: Path) -> None:
    import csv

    keys = [k for k in rows[0] if k != "best_params"] + ["best_params"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "best_params": json.dumps(r["best_params"])})
