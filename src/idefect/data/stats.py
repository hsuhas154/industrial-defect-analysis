"""Dataset statistics and sanity checks.

Before any model is trained, the dataset is audited for the problems that
silently corrupt results: unreadable files, unexpected sizes or channel counts,
exact duplicate images (which can leak between train and test), class
imbalance, and split leakage. The result is a JSON-serialisable report.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

from idefect.data.cv_ops import intensity_stats
from idefect.data.datasets import DatasetSplit, Sample, pixel_hash


def audit_samples(samples: Sequence[Sample], class_names: Sequence[str]) -> Dict:
    """Read every image once and collect per-class statistics and problems."""
    per_class_counts: Counter = Counter()
    shapes: Counter = Counter()
    dtypes: Counter = Counter()
    channels: Counter = Counter()
    extensions: Counter = Counter()
    unreadable: List[str] = []
    hashes: Dict[str, List[str]] = defaultdict(list)
    intensity: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for path, label in samples:
        cls = class_names[label]
        per_class_counts[cls] += 1
        extensions[Path(path).suffix.lower()] += 1
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            unreadable.append(str(path))
            continue
        shapes[f"{img.shape[0]}x{img.shape[1]}"] += 1
        dtypes[str(img.dtype)] += 1
        channels[1 if img.ndim == 2 else img.shape[2]] += 1
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hashes[pixel_hash(gray)].append(str(path))
        intensity[cls].append(intensity_stats(gray))

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]

    per_class_intensity = {}
    for cls, rows in intensity.items():
        per_class_intensity[cls] = {
            key: {
                "mean": float(np.mean([r[key] for r in rows])),
                "std": float(np.std([r[key] for r in rows])),
            }
            for key in ("mean", "std", "rms_contrast")
        }

    counts = [per_class_counts.get(c, 0) for c in class_names]
    imbalance = (max(counts) / min(counts)) if counts and min(counts) > 0 else None

    return {
        "n_images": len(samples),
        "n_classes": len(class_names),
        "class_names": list(class_names),
        "per_class_counts": {c: per_class_counts.get(c, 0) for c in class_names},
        "imbalance_ratio_max_over_min": imbalance,
        "image_shapes": dict(shapes),
        "dtypes": dict(dtypes),
        "channels": {str(k): v for k, v in channels.items()},
        "extensions": dict(extensions),
        "unreadable_files": unreadable,
        "duplicate_groups": duplicate_groups,
        "n_duplicate_images": sum(len(g) - 1 for g in duplicate_groups),
        "per_class_intensity": per_class_intensity,
    }


def audit_split(split: DatasetSplit, duplicate_groups: Sequence[Sequence[str]] = ()) -> Dict:
    """Check that a split is disjoint, stratified and free of duplicate leakage."""
    names = split.class_names
    parts = {"train": split.train, "val": split.val, "test": split.test}
    sets = {k: {p for p, _ in v} for k, v in parts.items()}

    overlaps = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }

    distribution = {
        k: {names[i]: sum(1 for _, y in v if y == i) for i in range(len(names))}
        for k, v in parts.items()
    }

    which = {p: k for k, s in sets.items() for p in s}
    leaking_groups = []
    for group in duplicate_groups:
        homes = {which.get(p) for p in group if p in which}
        if len(homes) > 1:
            leaking_groups.append(list(group))

    return {
        "sizes": {k: len(v) for k, v in parts.items()},
        "overlaps": overlaps,
        "is_disjoint": all(v == 0 for v in overlaps.values()),
        "class_distribution": distribution,
        "duplicate_groups_across_splits": leaking_groups,
    }


def problems(report: Dict, split_report: Dict | None = None) -> List[str]:
    """Human-readable list of issues found (empty list means all checks passed)."""
    issues: List[str] = []
    if report["unreadable_files"]:
        issues.append(f"{len(report['unreadable_files'])} unreadable image file(s)")
    if len(report["image_shapes"]) > 1:
        issues.append(f"mixed image sizes: {report['image_shapes']}")
    if len(report["channels"]) > 1 or "1" not in report["channels"]:
        issues.append(f"unexpected channel counts: {report['channels']}")
    leaking = split_report["duplicate_groups_across_splits"] if split_report else None
    if report["n_duplicate_images"] and (split_report is None or leaking):
        # Duplicates kept inside a single split cannot leak, so they only count
        # as a problem when no split is given or when a group crosses splits.
        issues.append(f"{report['n_duplicate_images']} exact duplicate image(s)")
    ratio = report["imbalance_ratio_max_over_min"]
    if ratio is not None and ratio > 1.5:
        issues.append(f"class imbalance ratio {ratio:.2f}")
    if split_report is not None:
        if not split_report["is_disjoint"]:
            issues.append(f"split overlap: {split_report['overlaps']}")
        if split_report["duplicate_groups_across_splits"]:
            issues.append(
                f"{len(split_report['duplicate_groups_across_splits'])} duplicate group(s) "
                "span more than one split (leakage)"
            )
    return issues
