"""One DAGM class through the shared-content check, in its own process.

Started by ``phase4_dagm_parallel`` (never needed by hand). Modes:

* ``run``: compute the check and write the cache file that
  ``scripts/phase4_dataset_report.py`` reads, exactly as the serial report
  would (same function, same arguments, same cache key).
* ``verify``: compute the check for a class that is already cached, write
  nothing to the cache, and compare the result with the cached one. Exit code
  0 means identical, 3 means different.
* ``smoke``: the first ``--limit`` images only, nothing cached; for testing
  the machinery.

Every image file is hashed and compared with ``data/dagm_manifest.sha256``
before it is used. The process dies with its parent, reports its stage and
progress to a status file, reserves memory through the shared ledger before
the memory-heavy steps and exits with a stack dump if it stops making
progress (see ``phase4_runtime``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from idefect.data.phase4_runtime import GB, Ledger, Status, die_with_parent

CLASS_SIZES = {c: (1150 if c <= 6 else 2300) for c in range(1, 11)}
TRANSFORMS = tuple(range(8))

# Memory estimates (bytes); measured on Class 7 (2,300 images of 512 x 512):
# 1.13 GB after loading, 5.41 GB peak while the E1 table is sorted.
BASE_RAM = int(0.45 * GB)            # interpreter, numpy, OpenCV
E1_SORT_PER_IMAGE = int(1.9e6)       # peak while hashing and sorting the reference runs
E1_HELD_PER_IMAGE = int(0.9e6)       # table kept while the helpers query it
HELPER_RAM = int(0.2 * GB)           # private memory of one forked helper
TORCH_HOST_RAM = int(1.2 * GB)       # CUDA libraries in host memory
GPU_CONTEXT = int(0.6 * GB)          # CUDA context and FFT plans
GPU_FINE = int(0.5 * GB)             # allocator cache of the fine and planted steps


def ram_estimates(n_images: int, image_bytes: int, helpers: int) -> dict:
    base = BASE_RAM + int(1.2 * image_bytes)
    e1 = base + max(n_images * E1_SORT_PER_IMAGE, n_images * E1_HELD_PER_IMAGE + helpers * HELPER_RAM)
    return {"base": base, "e1": e1, "gpu": base + TORCH_HOST_RAM}


def _first_difference(a, b, path="") -> list:
    """Up to a few paths where two JSON values differ."""
    if type(a) is not type(b):
        return [f"{path or '/'}: {type(a).__name__} vs {type(b).__name__}"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'cached' if k in a else 'recomputed'}")
            else:
                out += _first_difference(a[k], b[k], f"{path}/{k}")
            if len(out) >= 10:
                break
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: {len(a)} vs {len(b)} items"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _first_difference(x, y, f"{path}[{i}]")
            if len(out) >= 10:
                break
        return out
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cls", type=int, required=True)
    p.add_argument("--mode", choices=["run", "verify", "smoke"], required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--helpers", type=int, default=1)
    p.add_argument("--stall", type=float, default=300.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--config", default="configs/neu_baseline.yaml")
    a = p.parse_args(argv)
    die_with_parent()
    run_dir = Path(a.run_dir)
    st = Status(run_dir / "status" / f"{a.mode}_class{a.cls}.json", a.stall, cls=a.cls, mode=a.mode,
                attempt=a.attempt, helpers=a.helpers)
    try:
        return _work(a, run_dir, st)
    except BaseException as e:   # noqa: BLE001 - report every failure, then exit non-zero
        traceback.print_exc()
        st.finish("failed", error=f"{type(e).__name__}: {e}")
        return 2


def _work(a, run_dir: Path, st: Status) -> int:
    import cv2

    from idefect.config import load_config
    from idefect.data.archives import read_sha256_list
    from idefect.data.dagm import load_dagm
    from idefect.data.kolektor import read_image
    from idefect.data.phase4_leakage import _cache_file, content_check
    from idefect.data.similarity_screen import gpu_bytes

    cv2.setNumThreads(1)          # many processes share the CPU; OpenCV results do not depend on it
    led = Ledger(run_dir)
    cfg = load_config(a.config)
    dc, dup = cfg.dagm, cfg.data.duplicates
    c = a.cls
    expected = read_sha256_list(dc.manifest)
    sub = load_dagm(dc.root, [c])
    if len(sub) != CLASS_SIZES[c]:
        raise RuntimeError(f"Class{c} has {len(sub)} images, expected {CLASS_SIZES[c]}")
    if a.mode == "smoke":
        sub = sub[:a.limit]
    n = len(sub)
    est = ram_estimates(n, n * 512 * 512, a.helpers)
    on_wait = st.note
    led.set("ram", est["base"], f"class{c} images", on_wait)

    st.stage("load", n)
    root = Path(dc.root)
    images, ids, dig = [], [], []
    for k, s in enumerate(sub, 1):
        rel = Path(s.image).relative_to(root).as_posix()
        d = hashlib.sha256(Path(s.image).read_bytes()).hexdigest()
        if expected.get(rel) != d:
            raise RuntimeError(f"{rel} does not match {dc.manifest}")
        images.append(read_image(s.image))
        ids.append(s.id)
        dig.append(d)
        st.progress(k)

    hang = os.environ.get("IDEFECT_TEST_HANG", "") if a.mode == "smoke" and a.attempt == 1 else ""

    def progress(stage, done, total):
        if stage != st.stage_name:
            st.stage(stage, total)
        if stage == hang and done >= max(1, total // 3):
            print(f"test: simulating a hang in {stage}", flush=True)
            while True:      # a pure spin, like a stuck GPU wait: no progress, no system calls
                pass
        st.progress(done, total)

    @contextmanager
    def gate(stage, **info):
        if stage == "e1":
            waited = led.set("ram", est["e1"], f"class{c} E1 table", on_wait)
            st.stage("e1", n, f"waited {waited:.0f}s for RAM" if waited > 1 else "")
            yield
            led.set("ram", est["base"], f"class{c} images")
        elif stage == "coarse":
            need = GPU_CONTEXT + gpu_bytes([t.shape for t in info["thumbs"]], info["transforms"])
            st.stage("wait_gpu", 0, f"needs {need / GB:.1f} GB GPU")
            led.set("ram", est["gpu"], f"class{c} CUDA host", on_wait)
            waited = led.set("gpu", need, f"class{c} coarse screen", on_wait)
            st.stage("coarse", n * (n - 1) // 2, f"waited {waited:.0f}s for the GPU" if waited > 1 else "")
            yield
            import torch

            torch.cuda.empty_cache()
            led.set("gpu", GPU_CONTEXT + GPU_FINE, f"class{c} fine checks")
        else:
            yield

    tag = f"smoke_class{c}" if a.mode == "smoke" else f"dagm_class{c}"
    cache_dir = dup.cache_dir if a.mode == "run" else None
    t0 = time.time()
    res = content_check(images, ids, dig, dc.screen, dup, TRANSFORMS, tag, cache_dir, seed=c,
                        verbose=True, progress=progress, gate=gate, e1_workers=a.helpers)
    distinct_top = res["top_distinct"][0]["fine"] if res["top_distinct"] else None
    summary = {"n_images": res["n_images"], "exact": len(res["exact"]),
               "same_or_review": len(res["pairs_same_or_review"]),
               "review": sum(p["kind"] == "review" for p in res["pairs_same_or_review"]),
               "n_candidates": res["n_candidates"], "max_fine_distinct": res["max_fine_distinct"],
               "top_distinct": distinct_top,
               "planted_fine_min": min(v["fine_min"] for v in res["planted"].values()),
               "calibration_ok": res["calibration_ok"], "seconds": round(time.time() - t0, 1)}
    code = 0
    if a.mode == "verify":
        cache = _cache_file(dup.cache_dir, tag, dig, dc.screen, dup, TRANSFORMS)
        cached = json.loads(cache.read_text())
        diffs = _first_difference(cached, json.loads(json.dumps(res)))
        summary.update(identical=not diffs, differences=diffs, cache=str(cache),
                       cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
        (run_dir / f"verify_class{c}.json").write_text(json.dumps(summary, indent=1) + "\n")
        code = 0 if not diffs else 3
    elif a.mode == "run":
        summary["cache"] = str(_cache_file(dup.cache_dir, tag, dig, dc.screen, dup, TRANSFORMS))
        code = 0 if res["calibration_ok"] else 4
    print("[result] " + json.dumps(summary), flush=True)
    st.finish("done" if code == 0 else "verify_differs" if code == 3 else "calibration_failed",
              summary=summary)
    return code


if __name__ == "__main__":
    sys.exit(main())
