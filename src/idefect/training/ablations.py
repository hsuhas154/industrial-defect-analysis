"""Classical-branch ablations.

1. Preprocessing: does contrast enhancement or denoising help handcrafted
   features? Variants: plain resize, CLAHE (the default), CLAHE + median,
   CLAHE + Gaussian, median only.
2. Resolution: 64, 128 and 200 pixels (200 is native NEU).
3. HOG parameters: cell size, orientation bins, block size, normalization.

Every variant is re-tuned with the same stratified group k-fold search on the
training split. The cross-validation macro-F1 is the quantity compared, and
validation macro-F1 is a second check. The test split is never loaded here,
so no ablation result can be influenced by it.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score

from idefect.config import Config
from idefect.data.datasets import build_split, load_split_arrays
from idefect.features.registry import extract_many
from idefect.training.train_classical_ml import group_ids, tune

PREPROCESSING_VARIANTS = {
    "resize only": dict(clahe=False, denoise="none"),
    "CLAHE (default)": dict(clahe=True, denoise="none"),
    "CLAHE + median 3": dict(clahe=True, denoise="median", denoise_ksize=3),
    "CLAHE + Gaussian 3": dict(clahe=True, denoise="gaussian", denoise_ksize=3),
    "median 3 only": dict(clahe=False, denoise="median", denoise_ksize=3),
}

HOG_VARIANTS = [
    dict(pixels_per_cell=[8, 8], orientations=9, cells_per_block=[2, 2], block_norm="L2-Hys"),
    dict(pixels_per_cell=[16, 16], orientations=9, cells_per_block=[2, 2], block_norm="L2-Hys"),
    dict(pixels_per_cell=[32, 32], orientations=9, cells_per_block=[2, 2], block_norm="L2-Hys"),
    dict(pixels_per_cell=[16, 16], orientations=6, cells_per_block=[2, 2], block_norm="L2-Hys"),
    dict(pixels_per_cell=[16, 16], orientations=12, cells_per_block=[2, 2], block_norm="L2-Hys"),
    dict(pixels_per_cell=[16, 16], orientations=18, cells_per_block=[2, 2], block_norm="L2-Hys"),
    dict(pixels_per_cell=[16, 16], orientations=9, cells_per_block=[1, 1], block_norm="L2-Hys"),
    dict(pixels_per_cell=[16, 16], orientations=9, cells_per_block=[3, 3], block_norm="L2-Hys"),
    dict(pixels_per_cell=[16, 16], orientations=9, cells_per_block=[2, 2], block_norm="L1"),
    dict(pixels_per_cell=[16, 16], orientations=9, cells_per_block=[2, 2], block_norm="L2"),
]


def _evaluate(cfg: Config, split, g_train, feature_set: str, models: List[str], tag: Dict) -> List[Dict]:
    arrays = {p: load_split_arrays(getattr(split, p), cfg.preprocessing, cfg.data.image_size)
              for p in ("train", "val")}
    F = {p: extract_many(a[0], feature_set, cfg, n_jobs=cfg.classical_ml.n_jobs) for p, a in arrays.items()}
    rows = []
    for model in models:
        res = tune(model, F["train"], arrays["train"][1], g_train, cfg, verbose=False)
        est = res["estimator"]
        rows.append({
            **tag, "feature_set": feature_set, "model": model, "feature_dim": int(F["train"].shape[1]),
            "cv_macro_f1_mean": res["cv_macro_f1_mean"], "cv_macro_f1_std": res["cv_macro_f1_std"],
            "val_macro_f1": float(f1_score(arrays["val"][1], est.predict(F["val"]), average="macro")),
            "best_params": res["best_params"], "search_time_sec": res["search_time_sec"],
        })
        r = rows[-1]
        print(f"  {json.dumps(tag)} {feature_set:14s} {model:8s} dim={r['feature_dim']:5d} "
              f"CV {r['cv_macro_f1_mean']:.4f} +- {r['cv_macro_f1_std']:.4f}  val {r['val_macro_f1']:.4f}",
              flush=True)
    return rows


def run_ablations(cfg: Config, feature_sets: List[str], models: List[str],
                  hog_models: List[str], out_dir: Path) -> Dict[str, List[Dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, _, split, groups = build_split(cfg)
    g_train = group_ids(split.train, groups)
    results: Dict[str, List[Dict]] = {"preprocessing": [], "resolution": [], "hog": []}

    t0 = time.time()
    print("[ablation] preprocessing")
    for name, kw in PREPROCESSING_VARIANTS.items():
        c = copy.deepcopy(cfg)
        for k, v in kw.items():
            setattr(c.preprocessing, k, v)
        for fs in feature_sets:
            results["preprocessing"] += _evaluate(c, split, g_train, fs, models, {"preprocessing": name})

    print("[ablation] resolution")
    for size in (64, 128, 200):
        c = copy.deepcopy(cfg)
        c.data.image_size = size
        if size == 64:
            c.hog.pixels_per_cell = [8, 8]   # keep the same number of HOG cells
        if size == 200:
            c.hog.pixels_per_cell = [25, 25]
        for fs in feature_sets:
            results["resolution"] += _evaluate(c, split, g_train, fs, models, {"image_size": size})

    print("[ablation] HOG parameters")
    for hv in HOG_VARIANTS:
        c = copy.deepcopy(cfg)
        for k, v in hv.items():
            setattr(c.hog, k, v)
        tag = {"hog": f"cell {hv['pixels_per_cell'][0]}, bins {hv['orientations']}, "
                      f"block {hv['cells_per_block'][0]}, {hv['block_norm']}"}
        results["hog"] += _evaluate(c, split, g_train, "hog", hog_models, tag)

    for k, rows in results.items():
        with open(out_dir / f"ablation_{k}.json", "w") as fh:
            json.dump(rows, fh, indent=2)
    print(f"[ablation] done in {time.time() - t0:.0f}s")
    return results
