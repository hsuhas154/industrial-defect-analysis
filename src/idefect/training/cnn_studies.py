"""CNN sensitivity studies and the data-requirement curve.

1. Resolution: train at 64, 128 and 200 px (200 is native NEU).
2. Preprocessing: with and without CLAHE (the spec asks whether the
   preprocessing that helps classical features is beneficial or unnecessary
   for CNNs).
3. Data requirement: train on 10, 25, 50 and 100 percent of the training
   split, for CNNs and for classical models, to see which approach needs more
   data.

Like the classical ablations, these studies report **validation** macro-F1
only: the test split is never evaluated here, so nothing in these studies can
be influenced by it. Every setting is trained with several seeds.

Data subsets are drawn from the training split only, stratified by class, a
different subset per seed; validation and test stay exactly as they are.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from idefect.config import Config


def _summary(rows: List[Dict], keys: Sequence[str]) -> List[Dict]:
    out: Dict[tuple, List[float]] = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r["val_macro_f1"])
    return [{**dict(zip(keys, k)), "val_macro_f1_mean": float(np.mean(v)),
             "val_macro_f1_std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n_seeds": len(v)}
            for k, v in out.items()]


def _with(cfg: Config, arch: str, finetune: str) -> Config:
    c = copy.deepcopy(cfg)
    c.cnn.arch, c.cnn.finetune = arch, finetune
    return c


def _label(arch: str, ft: str) -> str:
    from idefect.models.registry import is_transfer

    return f"{arch} ({ft})" if is_transfer(arch) else arch


def _run_cnn(c: Config, seed: int, out_dir: Path, split_override=None) -> Dict:
    """Train one study run (validation only). Its checkpoint is deleted
    afterwards; only the per-epoch history is kept."""
    from idefect.training.train_cnn import _train_once

    m = _train_once(c, seed, out_dir, out_dir, verbose=False, split_override=split_override,
                    evaluate_test=False)
    for ckpt in Path(out_dir).glob("*.pt"):
        ckpt.unlink()
    return {"val_macro_f1": m["best_val_macro_f1"], "best_epoch": m["best_epoch"],
            "train_time_sec": m["train_time_sec"]}


def sensitivity_study(cfg: Config, models: Sequence[tuple], seeds: Sequence[int], out_dir: Path,
                      sizes: Sequence[int] = (64, 128, 200)) -> Dict:
    """``models``: (arch, finetune) pairs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for arch, ft in models:
        for size in sizes:
            for s in seeds:
                c = _with(cfg, arch, ft)
                c.data.image_size = size
                r = _run_cnn(c, s, out_dir / "runs" / f"{arch}_{ft}_res{size}_seed{s}")
                rows.append({"study": "resolution", "arch": arch, "finetune": ft, "variant": f"{size} px",
                             "seed": s, **r})
                print(f"[study] resolution {_label(arch, ft):30s} {size:4d}px seed {s}: val macro-F1 "
                      f"{r['val_macro_f1']:.4f} (best epoch {r['best_epoch']})", flush=True)
        for clahe in (False, True):
            if clahe == cfg.preprocessing.clahe and cfg.data.image_size in sizes:
                # the default setting is already covered by the resolution runs
                for r in [r for r in rows if r["study"] == "resolution" and r["arch"] == arch
                          and r["finetune"] == ft and r["variant"] == f"{cfg.data.image_size} px"]:
                    rows.append({**r, "study": "preprocessing", "variant": "CLAHE" if clahe else "no CLAHE"})
                continue
            for s in seeds:
                c = _with(cfg, arch, ft)
                c.preprocessing.clahe = clahe
                r = _run_cnn(c, s, out_dir / "runs" / f"{arch}_{ft}_clahe{int(clahe)}_seed{s}")
                rows.append({"study": "preprocessing", "arch": arch, "finetune": ft,
                             "variant": "CLAHE" if clahe else "no CLAHE", "seed": s, **r})
                print(f"[study] preprocessing {_label(arch, ft):30s} clahe={clahe} seed {s}: val macro-F1 "
                      f"{r['val_macro_f1']:.4f}", flush=True)
    result = {"rows": rows, "summary": _summary(rows, ("study", "arch", "finetune", "variant")),
              "note": "validation macro-F1 only; the test split is not used", "seconds": time.time() - t0}
    with open(out_dir / "cnn_sensitivity.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def subsample(samples: Sequence, fraction: float, seed: int) -> List:
    """Class-stratified random subset (at least 2 images per class)."""
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)
    out = []
    for c in sorted(by_class):
        items = by_class[c]
        k = max(2, int(round(fraction * len(items))))
        idx = rng.choice(len(items), size=min(k, len(items)), replace=False)
        out += [items[i] for i in sorted(idx)]
    return out


def data_curve(cfg: Config, cnn_models: Sequence[tuple], classical: Sequence[tuple],
               seeds: Sequence[int], out_dir: Path,
               fractions: Sequence[float] = (0.1, 0.25, 0.5, 1.0)) -> Dict:
    """Validation macro-F1 when training on a fraction of the training split.
    ``cnn_models``: (arch, finetune) pairs; ``classical``: (feature_set, model)."""
    from sklearn.metrics import f1_score

    from idefect.data.datasets import DatasetSplit, build_split, load_split_arrays
    from idefect.features.registry import extract_many
    from idefect.training.train_classical_ml import group_ids, tune

    out_dir.mkdir(parents=True, exist_ok=True)
    _, class_names, split, groups = build_split(cfg)
    rows = []
    t0 = time.time()
    feats: Dict[str, Dict] = {}
    for fs, _ in classical:
        if fs not in feats:
            arrays = {p: load_split_arrays(getattr(split, p), cfg.preprocessing, cfg.data.image_size)
                      for p in ("train", "val")}
            index = {p: i for i, (p, _) in enumerate(split.train)}
            feats[fs] = {"train": extract_many(arrays["train"][0], fs, cfg, n_jobs=cfg.classical_ml.n_jobs),
                         "val": extract_many(arrays["val"][0], fs, cfg, n_jobs=cfg.classical_ml.n_jobs),
                         "y_val": arrays["val"][1], "index": index}
    for frac in fractions:
        for s in seeds:
            sub = subsample(split.train, frac, s) if frac < 1 else list(split.train)
            part = DatasetSplit(train=sub, val=split.val, test=split.test, class_names=class_names)
            for arch, ft in cnn_models:
                r = _run_cnn(_with(cfg, arch, ft), s, out_dir / "runs" / f"{arch}_{ft}_frac{frac}_seed{s}",
                             split_override=part)
                rows.append({"branch": "cnn", "model": _label(arch, ft),
                             "fraction": frac, "n_train": len(sub), "seed": s, **r})
                print(f"[curve] {frac:5.2f} seed {s} {rows[-1]['model']:30s} n={len(sub):4d} val macro-F1 "
                      f"{r['val_macro_f1']:.4f}", flush=True)
            for fs, model in classical:
                F = feats[fs]
                idx = [F["index"][p] for p, _ in sub]
                y = np.array([lab for _, lab in sub])
                c = copy.deepcopy(cfg)
                c.classical_ml.cv_folds = min(cfg.classical_ml.cv_folds, int(np.bincount(y).min()))
                res = tune(model, F["train"][idx], y, group_ids(sub, groups), c, verbose=False)
                v = float(f1_score(F["y_val"], res["estimator"].predict(F["val"]), average="macro"))
                rows.append({"branch": "classical", "model": f"{fs}+{model}", "fraction": frac,
                             "n_train": len(sub), "seed": s, "val_macro_f1": v,
                             "train_time_sec": res["search_time_sec"]})
                print(f"[curve] {frac:5.2f} seed {s} {fs + '+' + model:30s} n={len(sub):4d} val macro-F1 {v:.4f}",
                      flush=True)
    result = {"rows": rows, "summary": _summary(rows, ("branch", "model", "fraction")),
              "note": "validation macro-F1 only; the test split is not used; training subsets are "
                      "class-stratified and drawn from the training split", "seconds": time.time() - t0}
    with open(out_dir / "data_curve.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return result
