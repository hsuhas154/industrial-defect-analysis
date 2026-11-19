"""Append-only experiment log.

Every training run appends one row to ``<logs_dir>/runs.csv`` with a timestamp,
the model kind, a short hash of the full config, the seed, the headline test
metrics and timings. Together with the per-run JSON metrics this gives the
reproducible experiment log the project requires.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict

FIELDS = [
    "timestamp",
    "model",
    "config_hash",
    "seed",
    "image_size",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "roc_auc_ovr_macro",
    "train_time_sec",
    "n_test",
    "config_path",
    "notes",
]


def config_hash(cfg: Any) -> str:
    """Stable 10-character hash of a (dataclass) config."""
    payload = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


def environment_info() -> Dict[str, str]:
    """Library versions and hardware, recorded alongside metrics."""
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "sklearn", "skimage", "cv2", "torch"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            continue
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def log_run(
    logs_dir: str | Path,
    model: str,
    cfg: Any,
    metrics: Dict[str, Any],
    config_path: str = "",
    notes: str = "",
) -> Path:
    """Append one row describing a finished run to ``runs.csv``."""
    path = Path(logs_dir) / "runs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "config_hash": config_hash(cfg),
        "seed": getattr(cfg, "seed", ""),
        "image_size": getattr(getattr(cfg, "data", None), "image_size", ""),
        "train_time_sec": metrics.get("train_time_sec", metrics.get("fit_time_sec", "")),
        "n_test": metrics.get("n_samples", ""),
        "config_path": config_path,
        "notes": notes,
    }
    for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc_ovr_macro"):
        v = metrics.get(k)
        row[k] = "" if v is None else f"{v:.6f}"
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return path
