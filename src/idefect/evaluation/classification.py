"""Classification metrics.

Per the project's evaluation policy, accuracy is never reported alone. Every
model reports accuracy, macro precision/recall/F1, per-class precision/recall/F1
and a confusion matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> Dict:
    """Compute the standard classification metric bundle.

    Returns a JSON-serialisable dict with overall scores, a per-class table and
    the confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))

    per_class_p = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_r = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_f = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class = {
        class_names[i]: {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f[i]),
        }
        for i in labels
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "n_samples": int(len(y_true)),
    }


def format_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> str:
    """Human-readable sklearn classification report."""
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        zero_division=0,
    )


def save_metrics(metrics: Dict, path: str | Path) -> None:
    """Write a metrics dict to JSON, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)


def summarize(metrics: Dict) -> str:
    """One-line summary of the headline scores."""
    return (
        f"acc={metrics['accuracy']:.4f}  "
        f"macro-P={metrics['macro_precision']:.4f}  "
        f"macro-R={metrics['macro_recall']:.4f}  "
        f"macro-F1={metrics['macro_f1']:.4f}  "
        f"(n={metrics['n_samples']})"
    )
