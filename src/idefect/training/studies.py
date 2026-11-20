"""Experiments that answer protocol questions rather than train a final model.

* :func:`cnn_epoch_study` - how long should the CNN train? Decided on
  validation macro-F1 only (several seeds), never on test.
* :func:`leakage_study` - how much does near-duplicate leakage inflate NEU
  scores? The same models are trained on (a) a plain stratified random split
  that ignores near-duplicates, as most NEU papers do, and (b) the
  group-aware split used everywhere else in this project. For (a) the share of
  test images that have a near-duplicate in the training split is reported.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from idefect.config import Config
from idefect.data.datasets import build_split, make_splits


def cnn_epoch_study(cfg: Config, epochs: Sequence[int], seeds: Sequence[int], out_dir: Path) -> List[Dict]:
    from idefect.training.train_cnn import _train_once

    rows = []
    for e in epochs:
        for s in seeds:
            c = copy.deepcopy(cfg)
            c.cnn.epochs = e
            c.output.logs_dir = str(out_dir / "logs")
            m = _train_once(c, s, out_dir / f"epochs{e}_seed{s}", out_dir, verbose=False,
                            evaluate_test=False)
            rows.append({"epochs": e, "seed": s, "val_macro_f1": m["best_val_macro_f1"],
                         "best_epoch": m["best_epoch"], "train_time_sec": m["train_time_sec"]})
            print(f"[epochs={e:3d} seed={s}] best val macro-F1 {m['best_val_macro_f1']:.4f} "
                  f"(epoch {m['best_epoch']}, {m['train_time_sec']:.0f}s, {m['device']})", flush=True)
    summary = []
    for e in epochs:
        v = np.array([r["val_macro_f1"] for r in rows if r["epochs"] == e])
        summary.append({"epochs": e, "val_macro_f1_mean": float(v.mean()), "val_macro_f1_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0})
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "epoch_study.json", "w") as fh:
        json.dump({"runs": rows, "summary": summary,
                   "note": "chosen on validation macro-F1 only; the test split is not used"}, fh, indent=2)
    return summary


def leakage_rate(split, groups: Sequence[Sequence[str]]) -> Dict:
    """Share of test (and val) images that share a near-duplicate group with a training image."""
    gid = {p: k for k, g in enumerate(groups) for p in g}
    train_groups = {gid[p] for p, _ in split.train if p in gid}
    out = {}
    for part in ("val", "test"):
        items = getattr(split, part)
        leaked = [p for p, _ in items if p in gid and gid[p] in train_groups]
        out[part] = {"n": len(items), "leaked": len(leaked), "fraction": len(leaked) / max(1, len(items))}
    return out


def leakage_study(cfg: Config, classical: Sequence[tuple], cnn_seeds: Sequence[int], out_dir: Path) -> Dict:
    """Train on a naive random split and on the group-aware split; compare test scores.

    ``classical`` is a list of (feature_set, model) pairs to evaluate.
    """
    from sklearn.metrics import f1_score

    from idefect.data.datasets import load_split_arrays
    from idefect.features.registry import extract_many
    from idefect.training.train_classical_ml import group_ids, tune
    from idefect.training.train_cnn import _train_once

    # build_split verifies the dataset manifest and returns the cached
    # near-duplicate groups; the group-aware split below is the same one it builds.
    samples, class_names, _, groups = build_split(cfg)
    splits = {
        "random (ignores near-duplicates)": make_splits(samples, class_names, cfg.data.train_frac,
                                                        cfg.data.val_frac, cfg.data.test_frac,
                                                        cfg.data.split_seed, groups=None),
        "group-aware (this project)": make_splits(samples, class_names, cfg.data.train_frac,
                                                  cfg.data.val_frac, cfg.data.test_frac,
                                                  cfg.data.split_seed, groups=groups),
    }
    result: Dict = {"leakage": {k: leakage_rate(s, groups) for k, s in splits.items()}, "models": []}
    for name, lr in result["leakage"].items():
        print(f"[leakage] {name}: {lr['test']['leaked']}/{lr['test']['n']} test images have a "
              f"near-duplicate in train ({lr['test']['fraction'] * 100:.1f}%)")

    for split_name, split in splits.items():
        arrays = {p: load_split_arrays(getattr(split, p), cfg.preprocessing, cfg.data.image_size)
                  for p in ("train", "test")}
        # CV groups: honest grouping for the group-aware split, none for the naive one.
        g = group_ids(split.train, groups if "group" in split_name else [])
        for fs, model in classical:
            F = {p: extract_many(a[0], fs, cfg, n_jobs=cfg.classical_ml.n_jobs) for p, a in arrays.items()}
            res = tune(model, F["train"], arrays["train"][1], g, cfg, verbose=False)
            f1 = float(f1_score(arrays["test"][1], res["estimator"].predict(F["test"]), average="macro"))
            result["models"].append({"split": split_name, "model": f"{fs}+{model}", "test_macro_f1": f1})
            print(f"[leakage] {split_name:34s} {fs}+{model:14s} test macro-F1 {f1:.4f}", flush=True)

        f1s = []
        for s in cnn_seeds:
            c = copy.deepcopy(cfg)
            c.output.logs_dir = str(out_dir / "logs")
            m = _train_once(c, s, out_dir / ("random" if "random" in split_name else "group") / f"seed{s}",
                            out_dir, verbose=False, split_override=split)
            f1s.append(m["macro_f1"])
            print(f"[leakage] {split_name:34s} custom_cnn seed={s}   test macro-F1 {m['macro_f1']:.4f} "
                  f"(best epoch {m['best_epoch']}, {m['train_time_sec']:.0f}s, {m['device']})", flush=True)
        result["models"].append({"split": split_name, "model": "custom_cnn",
                                 "test_macro_f1": float(np.mean(f1s)),
                                 "test_macro_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                                 "seeds": list(cnn_seeds)})
        print(f"[leakage] {split_name:34s} custom_cnn          test macro-F1 {np.mean(f1s):.4f} "
              f"+- {np.std(f1s, ddof=1) if len(f1s) > 1 else 0:.4f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "leakage_study.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return result
