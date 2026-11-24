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

from idefect.visualization.plots import _ensure_parent, plt

README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"

CLASSIFIER_ORDER = ["decision_tree", "logreg", "knn", "svm", "random_forest", "xgboost"]


def display(m: Dict) -> str:
    """Human-readable model name stored by the trainers (falls back to the id)."""
    return m.get("display_name") or m["model"]


def _order_key(m: Dict):
    if m.get("branch") == "cnn" or m["model"].startswith("custom_cnn"):
        from idefect.models.registry import ARCH_ORDER

        arch = m.get("arch", m["model"])
        tune = ("frozen", "partial", "full").index(m["finetune"]) if m.get("finetune") else 0
        return (2, ARCH_ORDER.index(arch) if arch in ARCH_ORDER else len(ARCH_ORDER), tune, m["model"])
    clf = m.get("classifier") or m["model"].split("+")[-1]
    rank = CLASSIFIER_ORDER.index(clf) if clf in CLASSIFIER_ORDER else len(CLASSIFIER_ORDER)
    return (1, rank, 0, m["model"])

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
    total = m.get("n_params_total", m.get("n_params"))  # n_params counts trainable weights only
    if total is not None:
        trainable = m.get("n_params")
        if trainable is not None and trainable != total:
            return f"{total:,} ({trainable:,} trained)"
        return f"{total:,}"
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
    if any(r.get("branch") == "cnn" for r in rows):
        note += (" For CNNs the metric columns are the seed kept by validation macro-F1 (the saved model);"
                 " the spread column gives the test macro-F1 mean +- std over all seeds. For classical"
                 " models the spread column is the cross-validation score on the training split.")
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
    """Horizontal bars, one row per model, sorted by test macro-F1.

    CNNs show the mean over their seeds with a standard-deviation bar (the
    honest number for a model family); classical models are deterministic and
    show their single score. Colour marks the family.
    """
    from matplotlib.patches import Patch

    def score(m):
        ss = (m.get("seed_summary") or {}).get("test_macro_f1")
        return (ss["mean"], ss["std"]) if ss else (m["macro_f1"], 0.0)

    def family(m):
        if m.get("branch") != "cnn" and not m["model"].startswith("custom_cnn"):
            return "classical"
        return "ImageNet CNN" if m.get("finetune") else "CNN from scratch"

    colors = {"classical": "tab:orange", "CNN from scratch": "tab:blue", "ImageNet CNN": "tab:green"}
    items = sorted(rows, key=lambda m: score(m)[0])
    fig, ax = plt.subplots(figsize=(9.5, 0.36 * len(items) + 1.6), layout="constrained")
    y = np.arange(len(items))
    vals = [score(m)[0] for m in items]
    errs = [score(m)[1] for m in items]
    ax.barh(y, vals, xerr=errs, capsize=2, color=[colors[family(m)] for m in items], height=0.7)
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + e + 0.002, yi, f"{v:.4f}", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels([display(m) for m in items], fontsize=8)
    lo = min(v - e for v, e in zip(vals, errs))
    ax.set_xlim(max(0.0, lo - 0.02), 1.012)
    ax.set(xlabel="test macro-F1 (CNNs: mean +- std over seeds; classical: single deterministic run)",
           title=f"Model comparison on NEU (held-out test split, n = {items[0].get('n_samples', '?')})")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in colors.items()], fontsize=8, loc="lower right")
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
        w.writerow(["model"] + [k for k, _ in COLUMNS] + ["train_time_sec", "n_params_trainable",
                                                          "n_params_total", "feature_dim"])
        for m in rows:
            w.writerow([m["model"]] + [m.get(k) for k, _ in COLUMNS] + [
                m.get("train_time_sec", m.get("fit_time_sec")), m.get("n_params"),
                m.get("n_params_total", m.get("n_params")), m.get("feature_dim")])

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
