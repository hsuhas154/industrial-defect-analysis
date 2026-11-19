"""Classification metrics.

Per the project's evaluation policy, accuracy is never reported alone. Every
model reports accuracy, macro precision/recall/F1, per-class precision/recall/F1
and a confusion matrix.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    y_proba: Optional[np.ndarray] = None,
) -> Dict:
    """Compute the standard classification metric bundle.

    Returns a JSON-serialisable dict with overall scores, a per-class table and
    the confusion matrix. When ``y_proba`` (shape ``(N, K)``) is given, the
    one-vs-rest ROC-AUC is added, both macro-averaged and per class.
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

    auc_macro, auc_per_class = None, None
    if y_proba is not None:
        auc_macro, auc_per_class = _ovr_roc_auc(y_true, np.asarray(y_proba), labels)
        for i in labels:
            per_class[class_names[i]]["roc_auc"] = (
                None if auc_per_class is None else auc_per_class[i]
            )

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
        "roc_auc_ovr_macro": auc_macro,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "n_samples": int(len(y_true)),
    }


def _ovr_roc_auc(y_true: np.ndarray, y_proba: np.ndarray, labels: List[int]):
    """One-vs-rest ROC-AUC. Returns ``(macro, per_class)`` or ``(None, None)``.

    AUC is undefined when a class is absent from ``y_true`` (no positives), so
    the score is skipped rather than reported misleadingly in that case.
    """
    if y_proba.ndim != 2 or y_proba.shape[1] != len(labels):
        return None, None
    if len(np.unique(y_true)) != len(labels):
        return None, None
    per_class = []
    for i in labels:
        per_class.append(float(roc_auc_score((y_true == i).astype(int), y_proba[:, i])))
    return float(np.mean(per_class)), per_class


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


def save_predictions(
    path: str | Path,
    image_paths: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    y_proba: Optional[np.ndarray] = None,
) -> Path:
    """Write per-image test predictions to CSV for error analysis.

    Columns: ``path, true, pred, correct, confidence`` and, if probabilities are
    available, one ``p_<class>`` column per class.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["path", "true", "pred", "correct", "confidence"]
    if y_proba is not None:
        header += [f"p_{c}" for c in class_names]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i, (p, t, q) in enumerate(zip(image_paths, y_true, y_pred)):
            row = [p, class_names[int(t)], class_names[int(q)], int(t == q)]
            if y_proba is not None:
                row.append(f"{float(y_proba[i][int(q)]):.6f}")
                row += [f"{float(v):.6f}" for v in y_proba[i]]
            else:
                row.append("")
            w.writerow(row)
    return path


def load_predictions(path: str | Path) -> List[Dict[str, str]]:
    """Read a predictions CSV written by :func:`save_predictions`."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def summarize(metrics: Dict) -> str:
    """One-line summary of the headline scores."""
    auc = metrics.get("roc_auc_ovr_macro")
    auc_txt = f"  AUC={auc:.4f}" if auc is not None else ""
    return (
        f"acc={metrics['accuracy']:.4f}  "
        f"macro-P={metrics['macro_precision']:.4f}  "
        f"macro-R={metrics['macro_recall']:.4f}  "
        f"macro-F1={metrics['macro_f1']:.4f}"
        f"{auc_txt}  (n={metrics['n_samples']})"
    )
