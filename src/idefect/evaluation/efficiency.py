"""Efficiency benchmark: what does each model cost to run?

For every trained artifact in the results directory:

* parameters (trainable and total) and size on disk;
* preprocessing time per image (resize + CLAHE, shared by every model);
* single-image latency on the CPU and, for CNNs, on the GPU: the median over
  ``repeats`` timed calls after ``warmup`` untimed ones, so one-off costs
  (lazy initialization, cuDNN autotuning, caches) are excluded;
* throughput at batch 64 (images per second), CPU and GPU;
* peak GPU memory for batch-64 inference and for one batch-64 training step
  (forward, backward and optimizer update).

Classical models run on the CPU only; their "model" time is feature
extraction plus prediction, which is how they are used. Timings depend on the
machine; the environment is recorded with the results. Run it with nothing
else on the GPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np


def _median_ms(fn: Callable[[], None], warmup: int, repeats: int, sync: Callable[[], None]) -> float:
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def _images(cfg, n: int) -> List[np.ndarray]:
    """``n`` preprocessed training images (any split would do for timing)."""
    from idefect.data.datasets import build_split
    from idefect.data.preprocessing import load_and_preprocess

    _, _, split, _ = build_split(cfg)
    return [load_and_preprocess(p, cfg.preprocessing, cfg.data.image_size) for p, _ in split.train[:n]]


def preprocessing_ms(cfg, warmup: int = 5, repeats: int = 50) -> float:
    from idefect.data.datasets import build_split
    from idefect.data.preprocessing import load_grayscale, preprocess_image

    _, _, split, _ = build_split(cfg)
    raw = load_grayscale(split.train[0][0])
    return _median_ms(lambda: preprocess_image(raw, cfg.preprocessing, cfg.data.image_size),
                      warmup, repeats, lambda: None)


def benchmark_cnn(path: Path, imgs: List[np.ndarray], warmup: int = 10, repeats: int = 50) -> Dict:
    import torch

    from idefect.inference.predict import Predictor

    pred = Predictor(path)
    model = pred.model
    x1 = torch.from_numpy(np.stack(imgs[:1]))[:, None].float()
    x64 = torch.from_numpy(np.stack(imgs[:64]))[:, None].float()
    out = {"model": pred.kind, "branch": "cnn",
           "params_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
           "params_total": int(sum(p.numel() for p in model.parameters())),
           "size_mb": round(path.stat().st_size / 2**20, 3), "cpu_threads": torch.get_num_threads()}

    model = model.cpu().eval()
    n = len(x64)
    with torch.no_grad():
        out["cpu_latency_ms"] = _median_ms(lambda: model(x1), warmup, repeats, lambda: None)
        t = _median_ms(lambda: model(x64), 3, 10, lambda: None)
    out["cpu_throughput_ips"] = n / (t / 1000.0)

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        sync = torch.cuda.synchronize
        model = model.to(dev).eval()
        g1, g64 = x1.to(dev), x64.to(dev)
        with torch.no_grad():
            out["gpu_latency_ms"] = _median_ms(lambda: model(g1), warmup, repeats, sync)
            t = _median_ms(lambda: model(g64), warmup, repeats, sync)
            out["gpu_throughput_ips"] = n / (t / 1000.0)
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            model(g64)
            sync()
            out["gpu_infer_peak_mb"] = (torch.cuda.max_memory_allocated() - base) / 2**20
        params = [p for p in model.parameters() if p.requires_grad]
        if params:
            opt = torch.optim.Adam(params, lr=1e-4)
            model.train()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            y = torch.zeros(len(g64), dtype=torch.long, device=dev)
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(g64), y).backward()
            opt.step()
            sync()
            out["gpu_train_peak_mb"] = (torch.cuda.max_memory_allocated() - base) / 2**20
            del opt
        model.cpu()
        torch.cuda.empty_cache()
    return out


def benchmark_classical(path: Path, imgs: List[np.ndarray], warmup: int = 3, repeats: int = 30) -> Dict:
    from idefect.features.registry import extract_one
    from idefect.inference.predict import Predictor

    pred = Predictor(path)
    est, fs, fcfg = pred.model, pred.feature_set, pred._feature_cfg
    one = lambda: est.predict(extract_one(imgs[0], fs, fcfg)[None])  # noqa: E731
    batch_imgs = imgs[:64]

    def batch():
        est.predict(np.stack([extract_one(im, fs, fcfg) for im in batch_imgs]))

    t = _median_ms(batch, 1, 5, lambda: None)
    return {"model": pred.kind, "branch": "classical", "size_mb": round(path.stat().st_size / 2**20, 3),
            "cpu_latency_ms": _median_ms(one, warmup, repeats, lambda: None),
            "cpu_throughput_ips": len(batch_imgs) / (t / 1000.0)}


def run_benchmark(cfg, results_dir: Path, out_dir: Path, models: List[str] = None) -> List[Dict]:
    from idefect.experiment_log import environment_info

    imgs = _images(cfg, 64)
    pre = preprocessing_ms(cfg)
    rows = []
    paths = sorted(results_dir.glob("*.joblib")) + sorted(results_dir.glob("*.pt"))
    for p in paths:
        if models and p.stem not in models:
            continue
        r = benchmark_cnn(p, imgs) if p.suffix == ".pt" else benchmark_classical(p, imgs)
        r["preprocess_ms"] = pre
        mfile = results_dir / f"{p.stem}_metrics.json"
        if mfile.exists():
            m = json.loads(mfile.read_text())
            ss = m.get("seed_summary", {})
            r.update(display_name=m.get("display_name", p.stem), test_macro_f1=m["macro_f1"],
                     test_macro_f1_mean=ss.get("test_macro_f1", {}).get("mean"),
                     test_macro_f1_std=ss.get("test_macro_f1", {}).get("std"),
                     train_time_sec=ss.get("train_time_sec", {}).get("mean", m.get("train_time_sec",
                                                                                    m.get("fit_time_sec"))))
        rows.append(r)
        print(f"[bench] {r['model']:34s} cpu {r['cpu_latency_ms']:8.2f} ms"
              + (f"  gpu {r['gpu_latency_ms']:6.2f} ms" if "gpu_latency_ms" in r else ""), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "efficiency.json", "w") as fh:
        json.dump({"rows": rows, "environment": environment_info(), "image_size": cfg.data.image_size,
                   "preprocess_ms": pre}, fh, indent=2)
    return rows
