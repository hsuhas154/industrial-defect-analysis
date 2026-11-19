"""Error analysis from saved test predictions.

Reads a ``*_test_predictions.csv`` written by the trainers and produces:

* a gallery of misclassified test images, most confident mistakes first
  (those are the most informative: the model was sure and wrong);
* the most frequently confused (true, predicted) class pairs;
* a row-normalized confusion matrix figure (per-class recall on the diagonal).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from idefect.data.preprocessing import load_grayscale
from idefect.evaluation.classification import load_predictions
from idefect.visualization.plots import _ensure_parent, plot_confusion_matrix, plt


def misclassified(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Wrong predictions sorted by descending confidence (unknown confidence last)."""
    wrong = [r for r in rows if r["correct"] == "0"]

    def conf(r):
        try:
            return float(r["confidence"])
        except (TypeError, ValueError):
            return -1.0

    return sorted(wrong, key=conf, reverse=True)


def confused_pairs(rows: Sequence[Dict[str, str]]) -> List[Dict]:
    """Count of each (true, predicted) mistake, most frequent first."""
    c = Counter((r["true"], r["pred"]) for r in rows if r["correct"] == "0")
    return [{"true": t, "pred": p, "count": n} for (t, p), n in c.most_common()]


def confusion_from_rows(rows: Sequence[Dict[str, str]], class_names: Sequence[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(class_names)}
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for r in rows:
        cm[idx[r["true"]], idx[r["pred"]]] += 1
    return cm


def plot_misclassified_gallery(
    rows: Sequence[Dict[str, str]],
    out_path: str | Path,
    title: str,
    max_images: int = 24,
    cols: int = 6,
) -> Path | None:
    """Grid of misclassified images captioned with true, predicted and confidence."""
    wrong = misclassified(rows)[:max_images]
    if not wrong:
        return None
    n = len(wrong)
    cols = min(cols, n)
    grid_rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(grid_rows, cols, figsize=(2.1 * cols, 2.45 * grid_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, wrong):
        ax.imshow(load_grayscale(r["path"]), cmap="gray", vmin=0, vmax=255)
        conf = r.get("confidence") or ""
        conf_txt = f" ({float(conf):.2f})" if conf else ""
        ax.set_title(f"true: {r['true']}\npred: {r['pred']}{conf_txt}", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    out_path = _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def analyze_predictions(
    predictions_csv: str | Path,
    figures_dir: str | Path,
    class_names: Sequence[str] | None = None,
) -> Dict:
    """Run the full error analysis for one model's predictions file."""
    predictions_csv = Path(predictions_csv)
    model = predictions_csv.name.replace("_test_predictions.csv", "")
    rows = load_predictions(predictions_csv)
    if class_names is None:
        class_names = sorted({r["true"] for r in rows} | {r["pred"] for r in rows})
    figures_dir = Path(figures_dir)

    gallery = plot_misclassified_gallery(
        rows, figures_dir / f"{model}_misclassified.png",
        title=f"{model}: misclassified test images (most confident first)",
    )
    cm = confusion_from_rows(rows, class_names)
    plot_confusion_matrix(
        cm, class_names, figures_dir / f"{model}_confusion_matrix_normalized.png",
        title=f"{model} confusion matrix (row-normalized)", normalize=True,
    )
    n_wrong = sum(1 for r in rows if r["correct"] == "0")
    return {
        "model": model,
        "n_test": len(rows),
        "n_wrong": n_wrong,
        "error_rate": n_wrong / len(rows) if rows else 0.0,
        "confused_pairs": confused_pairs(rows),
        "most_confident_mistakes": [
            {k: r[k] for k in ("path", "true", "pred", "confidence")}
            for r in misclassified(rows)[:10]
        ],
        "gallery": str(gallery) if gallery else None,
    }
