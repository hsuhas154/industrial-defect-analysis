"""CNN-branch figures: Grad-CAM grids and cross-model comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from idefect.visualization.plots import plt


def _save(fig, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _overlay(ax, img: np.ndarray, cam: np.ndarray, title: str = "", fontsize: int = 7) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=fontsize)
    ax.axis("off")


def plot_gradcam_grid(images: np.ndarray, cams: np.ndarray, titles: Sequence[str], out_path,
                      title: str, cols: int = 4) -> Path:
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.75 * rows), squeeze=False,
                             layout="constrained")
    for i, ax in enumerate(axes.flat):
        if i < n:
            _overlay(ax, images[i], cams[i], titles[i])
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=10)
    return _save(fig, out_path)


def plot_gradcam_comparison(images: Sequence[np.ndarray], names: Sequence[str],
                            per_model: Dict[str, tuple], out_path) -> Path:
    """Rows: images; first column the input, then one Grad-CAM per model with
    its predicted class."""
    models = list(per_model)
    rows, cols = len(images), len(models) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(1.9 * cols, 2.0 * rows), squeeze=False,
                             layout="constrained")
    for r, (img, name) in enumerate(zip(images, names)):
        ax = axes[r, 0]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=7)
        ax.axis("off")
        for c, m in enumerate(models, start=1):
            cams, preds = per_model[m]
            _overlay(axes[r, c], img, cams[r], (m + "\n" if r == 0 else "") + f"pred {preds[r]}", fontsize=6)
    fig.suptitle("Grad-CAM across CNNs (test images, one per class, and Sc_44)", fontsize=10)
    return _save(fig, out_path)


def plot_accuracy_vs_cost(rows: List[Dict], x_key: str, x_label: str, out_path, title: str) -> Path:
    """Test macro-F1 (seed mean and standard deviation for CNNs) against a cost
    on a log x-axis. Points are numbered and named in a key beside the plot, so
    labels never overlap."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator

    pts = [r for r in rows if r.get(x_key) is not None and r.get("test_macro_f1") is not None]
    pts.sort(key=lambda r: -(r.get("test_macro_f1_mean") or r["test_macro_f1"]))
    fig, (ax, key) = plt.subplots(1, 2, figsize=(12.5, 5.6), width_ratios=[2.2, 1], layout="constrained")
    lines = []
    for i, r in enumerate(pts, start=1):
        y = r.get("test_macro_f1_mean") or r["test_macro_f1"]
        err = r.get("test_macro_f1_std") or 0
        color = "tab:blue" if r["branch"] == "cnn" else "tab:orange"
        ax.errorbar(r[x_key], y, yerr=err, fmt="o", color=color, capsize=2, ms=5)
        ax.annotate(str(i), (r[x_key], y), fontsize=7, xytext=(4, 2), textcoords="offset points")
        lines.append(f"{i:2d}  {r.get('display_name', r['model'])}  ({y:.4f})")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set(xlabel=x_label, ylabel="test macro-F1 (mean over seeds for CNNs)", title=title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(handles=[Line2D([], [], color="tab:blue", marker="o", ls="", label="CNN (bar: std over 5 seeds)"),
                       Line2D([], [], color="tab:orange", marker="o", ls="", label="classical (deterministic)")],
              loc="lower left", fontsize=8)
    key.axis("off")
    key.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=7.5, family="monospace",
             transform=key.transAxes)
    return _save(fig, out_path)


def _pretty(model: str) -> str:
    """Display name for a study label: 'arch' or 'arch (finetune)' or 'features+classifier'."""
    from idefect.models.registry import available, display_name

    arch, _, rest = model.partition(" (")
    if arch in available():
        return display_name(arch, rest.rstrip(")") or "full")
    names = {"logreg": "LogReg", "svm": "SVM", "knn": "k-NN", "decision_tree": "Decision Tree",
             "random_forest": "Random Forest", "xgboost": "XGBoost"}
    *feats, clf = model.split("+")
    return " + ".join(f.upper() for f in feats) + " + " + names.get(clf, clf)


def plot_data_curve(summary: List[Dict], n_by_fraction: Dict[float, int], out_path) -> Path:
    """Validation macro-F1 against the number of training images (mean and std over seeds)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0), layout="constrained")
    models = list(dict.fromkeys(s["model"] for s in summary))
    styles = {"cnn": "-", "classical": "--"}
    for m in models:
        rows = sorted((s for s in summary if s["model"] == m), key=lambda s: s["fraction"])
        x = [n_by_fraction[s["fraction"]] for s in rows]
        y = [s["val_macro_f1_mean"] for s in rows]
        e = [s["val_macro_f1_std"] for s in rows]
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, ls=styles.get(rows[0]["branch"], "-"),
                    label=_pretty(m))
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter

    ax.xaxis.set_major_locator(FixedLocator(sorted(n_by_fraction.values())))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set(xlabel="training images (log scale; 10, 25, 50, 100 percent of the training split)",
           ylabel="validation macro-F1 (mean +- std, 3 seeds)",
           title="How much data does each approach need? (validation split)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", title="solid: CNN, dashed: classical", title_fontsize=8)
    return _save(fig, out_path)


def plot_sensitivity(summary: List[Dict], study: str, variants: Sequence[str], out_path, title: str) -> Path:
    """Grouped bars: one group per model, one bar per variant (validation macro-F1)."""
    rows = [s for s in summary if s["study"] == study]
    models = list(dict.fromkeys((s["arch"], s["finetune"]) for s in rows))
    x = np.arange(len(models))
    w = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(8.5, 4.6), layout="constrained")
    for i, v in enumerate(variants):
        vals = [next((s["val_macro_f1_mean"] for s in rows if (s["arch"], s["finetune"]) == m and s["variant"] == v),
                     np.nan) for m in models]
        errs = [next((s["val_macro_f1_std"] for s in rows if (s["arch"], s["finetune"]) == m and s["variant"] == v),
                     0) for m in models]
        ax.bar(x + (i - (len(variants) - 1) / 2) * w, vals, w, yerr=errs, capsize=2, label=v)
    from idefect.models.registry import display_name

    ax.set_xticks(x)
    ax.set_xticklabels([display_name(a, f) for a, f in models], fontsize=8)
    lo = min(s["val_macro_f1_mean"] - s["val_macro_f1_std"] for s in rows)
    ax.set_ylim(max(0.0, lo - 0.02), 1.005)
    ax.set(ylabel="validation macro-F1 (mean +- std, 3 seeds)", title=title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    return _save(fig, out_path)
