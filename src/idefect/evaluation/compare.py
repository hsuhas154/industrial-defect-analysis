"""Collect per-model metrics into one comparison table.

Reads every ``*_metrics.json`` in the results directory and produces:

* ``summary.md``: a Markdown table (also usable to refresh the README);
* ``summary.csv``: the same numbers in machine-readable form;
* a grouped bar chart of accuracy, macro-F1 and ROC-AUC per model.

Numbers in the README are generated from these files, never typed by hand.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from idefect.visualization.plots import _ensure_parent, plt, tidy

README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"

CLASSIFIER_ORDER = ["decision_tree", "logreg", "knn", "svm", "random_forest", "xgboost"]


def display(m: Dict) -> str:
    """Human-readable model name stored by the trainers (falls back to the id)."""
    return m.get("display_name") or m["model"]


def _order_key(m: Dict):
    if m.get("branch") == "cnn" or m["model"].startswith("custom_cnn"):
        return (2, 0, m["model"])
    clf = m.get("classifier") or m["model"].split("+")[-1]
    rank = CLASSIFIER_ORDER.index(clf) if clf in CLASSIFIER_ORDER else len(CLASSIFIER_ORDER)
    return (1, rank, m["model"])

COLUMNS = [
    ("accuracy", "Accuracy"),
    ("macro_precision", "Macro P"),
    ("macro_recall", "Macro R"),
    ("macro_f1", "Macro F1"),
    ("roc_auc_ovr_macro", "ROC-AUC (OvR)"),
]


def collect_metrics(results_dir: str | Path) -> List[Dict]:
    """Load all ``*_metrics.json`` files.

    Classical models come first (ordered by classifier family, see
    ``CLASSIFIER_ORDER``), CNNs last; ties are broken by model id.
    """
    rows = []
    for p in sorted(Path(results_dir).glob("*_metrics.json")):
        with open(p) as fh:
            m = json.load(fh)
        m.setdefault("model", p.name.replace("_metrics.json", ""))
        rows.append(m)
    return sorted(rows, key=_order_key)


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def _time(m: Dict) -> str:
    t = m.get("train_time_sec", m.get("fit_time_sec"))
    if t is None:
        return "n/a"
    return f"{t:.3f}" if t < 10 else f"{t:.1f}"


def _spread(m: Dict) -> str:
    """Variability estimate: CV macro-F1 for classical models (std over folds on
    the training split), test macro-F1 over seeds for CNNs."""
    ss = m.get("seed_summary")
    if ss:
        t = ss["test_macro_f1"]
        return f"{t['mean']:.4f} +- {t['std']:.4f} (test, {len(ss['seeds'])} seeds)"
    if m.get("cv_macro_f1_mean") is not None:
        return f"{m['cv_macro_f1_mean']:.4f} +- {m['cv_macro_f1_std']:.4f} (CV)"
    return "n/a"


def _latency(m: Dict) -> str:
    v = m.get("latency_ms_per_image")
    return "n/a" if v is None else f"{v:.2f}"


def _params(m: Dict) -> str:
    if m.get("n_params") is not None:
        return f"{m['n_params']:,}"
    if m.get("feature_dim") is not None:
        return f"{m['feature_dim']:,}-d features"
    return "n/a"


def markdown_table(rows: List[Dict], include_time: bool = True) -> str:
    """Markdown results table.

    ``include_time=False`` drops the wall-clock training time, which varies
    from run to run; the README uses that form so a rerun that reproduces the
    same metrics leaves the README unchanged.
    """
    header = ["Model"] + [c[1] for c in COLUMNS] + ["Macro F1 spread"] + \
        (["Train time (s)", "Latency (ms/img)"] if include_time else []) + ["Size"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for m in rows:
        cells = [display(m)] + [_fmt(m.get(k)) for k, _ in COLUMNS] + [_spread(m)] + \
            ([_time(m), _latency(m)] if include_time else []) + [_params(m)]
        lines.append("| " + " | ".join(cells) + " |")
    n = {m.get("n_samples") for m in rows}
    note = f"\n\nAll scores on the same held-out test split (n = {n.pop()} images)." if len(n) == 1 else ""
    return "\n".join(lines) + note + "\n"


def per_class_f1_table(rows: List[Dict]) -> str:
    if not rows:
        return ""
    classes = rows[0]["class_names"]
    header = ["Class"] + [display(m) for m in rows]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for c in classes:
        cells = [c] + [_fmt(m["per_class"][c]["f1"]) for m in rows]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_comparison(rows: List[Dict], out_path: str | Path) -> Path:
    keys = [("accuracy", "accuracy"), ("macro_f1", "macro-F1"), ("roc_auc_ovr_macro", "ROC-AUC")]
    names = [display(m).replace(" + ", "\n+ ", 1) if len(display(m)) > 18 else display(m)
             for m in rows]
    x = np.arange(len(rows))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 1.9 * len(rows) + 2.0), 4.4))
    for i, (k, label) in enumerate(keys):
        vals = [m.get(k) or 0.0 for m in rows]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center",
                    va="bottom", fontsize=6.5)
    lo = min([m.get("accuracy") or 1.0 for m in rows] + [m.get("macro_f1") or 1.0 for m in rows])
    ax.set_ylim(max(0.0, lo - 0.1), 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set(ylabel="score (test split)", title="Model comparison on NEU")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", alpha=0.3)
    tidy(fig)
    out_path = _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_summary(results_dir: str | Path, figures_dir: str | Path) -> Dict[str, Path]:
    """Write summary.md, summary.csv and the comparison chart."""
    rows = collect_metrics(results_dir)
    if not rows:
        raise FileNotFoundError(f"no *_metrics.json files in {results_dir}; train a model first")
    results_dir = Path(results_dir)
    md = "## Overall\n\n" + markdown_table(rows) + "\n## Per-class F1\n\n" + per_class_f1_table(rows)
    md_path = results_dir / "summary.md"
    md_path.write_text(md)

    csv_path = results_dir / "summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model"] + [k for k, _ in COLUMNS] + ["train_time_sec", "n_params", "feature_dim"])
        for m in rows:
            w.writerow([m["model"]] + [m.get(k) for k, _ in COLUMNS] + [
                m.get("train_time_sec", m.get("fit_time_sec")), m.get("n_params"),
                m.get("feature_dim")])

    chart = plot_comparison(rows, Path(figures_dir) / "model_comparison.png")
    return {"markdown": md_path, "csv": csv_path, "chart": chart}


def update_readme(readme: str | Path, table_md: str) -> bool:
    """Replace the text between the RESULTS markers in the README.

    Returns True if the README changed.
    """
    readme = Path(readme)
    text = readme.read_text()
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.S)
    if not pattern.search(text):
        raise ValueError(f"{readme} has no {README_START} ... {README_END} block")
    new = pattern.sub(README_START + "\n" + table_md.strip() + "\n" + README_END, text)
    if new != text:
        readme.write_text(new)
        return True
    return False
