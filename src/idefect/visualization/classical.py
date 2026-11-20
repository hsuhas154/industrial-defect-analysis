"""Figures for the classical branch.

* feature set x classifier heatmaps (CV and test macro-F1);
* one-vs-rest ROC and precision-recall curves per final model;
* 2-D feature-space projections (PCA and t-SNE) per descriptor;
* LBP code images and per-class mean LBP histograms;
* per-class distributions of GLCM statistics;
* decision-tree depth versus train / cross-validation macro-F1 (overfitting);
* Random Forest and XGBoost feature importances;
* ablation bar charts (preprocessing, resolution, HOG parameters).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from idefect.visualization.plots import _ensure_parent, plt, tidy

CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _save(fig, out_path) -> Path:
    tidy(fig)
    out_path = _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_grid_heatmap(rows: List[Dict], key: str, out_path, title: str) -> Path:
    fsets = list(dict.fromkeys(r["feature_set"] for r in rows))
    models = list(dict.fromkeys(r["model"] for r in rows))
    M = np.full((len(fsets), len(models)), np.nan)
    for r in rows:
        M[fsets.index(r["feature_set"]), models.index(r["model"])] = r[key]
    fig, ax = plt.subplots(figsize=(1.45 * len(models) + 2.5, 0.62 * len(fsets) + 1.6))
    im = ax.imshow(M, cmap="viridis", vmin=np.nanmin(M), vmax=np.nanmax(M), aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(fsets)))
    ax.set_yticklabels([f.upper() for f in fsets], fontsize=8)
    for i in range(len(fsets)):
        for j in range(len(models)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < np.nanmean(M) else "black")
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(title, fontsize=10)
    return _save(fig, out_path)


def plot_roc_pr(y_true: np.ndarray, proba: np.ndarray, class_names: Sequence[str], out_path,
                title: str) -> Path:
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.6))
    for k, name in enumerate(class_names):
        yk = (y_true == k).astype(int)
        fpr, tpr, _ = roc_curve(yk, proba[:, k])
        a1.plot(fpr, tpr, color=CLASS_COLORS[k % 6], lw=1.4, label=f"{name} (AUC {auc(fpr, tpr):.3f})")
        pr, rc, _ = precision_recall_curve(yk, proba[:, k])
        a2.plot(rc, pr, color=CLASS_COLORS[k % 6], lw=1.4,
                label=f"{name} (AP {average_precision_score(yk, proba[:, k]):.3f})")
    n_neg_min = max(1, int(min((y_true != k).sum() for k in range(len(class_names)))))
    floor = 0.5 / n_neg_min
    grid = np.logspace(np.log10(floor), 0, 50)
    a1.plot(grid, grid, "k:", lw=0.8, label="chance (random classifier)")
    a1.set_xscale("log")
    a1.set_xlim(floor, 1.0)
    a1.set(xlabel="false positive rate (log scale)", ylabel="true positive rate",
           title="ROC (one-vs-rest)")
    a2.set(xlabel="recall", ylabel="precision", title="Precision-recall (one-vs-rest)")
    # Legends go below each panel so they never cover a curve or the dotted
    # chance line (a random classifier), whatever the model.
    for a in (a1, a2):
        a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
                 frameon=False)
        a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    return _save(fig, out_path)


def plot_projections(features: Dict[str, np.ndarray], y: np.ndarray, class_names: Sequence[str],
                     out_path, seed: int = 0) -> Path:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    names = list(features)
    fig, axes = plt.subplots(2, len(names), figsize=(3.6 * len(names), 7.0), squeeze=False)
    for j, n in enumerate(names):
        X = StandardScaler().fit_transform(features[n])
        pca = PCA(n_components=2, random_state=seed).fit(X)
        P = pca.transform(X)
        Xr = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X)
        T = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(Xr)
        for row, Z, label in ((0, P, f"PCA ({pca.explained_variance_ratio_.sum() * 100:.0f}% var)"),
                              (1, T, "t-SNE")):
            ax = axes[row, j]
            for k, cname in enumerate(class_names):
                m = y == k
                ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.6, color=CLASS_COLORS[k % 6], label=cname)
            ax.set_title(f"{n.upper()}: {label}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].legend(fontsize=7, markerscale=3, loc="best")
    fig.suptitle("Feature-space projections (all images, coloured by class)", fontsize=10)
    return _save(fig, out_path)


def plot_lbp(images: Sequence[np.ndarray], labels: Sequence[int], class_names: Sequence[str],
             lbp_cfg, out_path) -> Path:
    from idefect.features.lbp import lbp_codes, n_bins

    P, R = int(lbp_cfg.points_radii[0][0]), float(lbp_cfg.points_radii[0][1])
    bins = n_bins(P, lbp_cfg.method)
    K = len(class_names)
    fig = plt.figure(figsize=(2.1 * K, 6.8))
    gs = fig.add_gridspec(3, K, height_ratios=[1, 1, 1.15])
    hists = {k: [] for k in range(K)}
    for img, y in zip(images, labels):
        c = lbp_codes(img, P, R, lbp_cfg.method)
        hists[y].append(np.bincount(c.astype(int).ravel(), minlength=bins)[:bins] / c.size)
    shown = set()
    for img, y in zip(images, labels):
        if y in shown:
            continue
        shown.add(y)
        a = fig.add_subplot(gs[0, y])
        a.imshow(img, cmap="gray")
        a.set_title(class_names[y], fontsize=8)
        a.axis("off")
        b = fig.add_subplot(gs[1, y])
        b.imshow(lbp_codes(img, P, R, lbp_cfg.method), cmap="nipy_spectral")
        b.set_title(f"LBP codes P={P}, R={R:g}", fontsize=7)
        b.axis("off")
    ax = fig.add_subplot(gs[2, :])
    w = 0.8 / K
    for k in range(K):
        if hists[k]:
            ax.bar(np.arange(bins) + (k - (K - 1) / 2) * w, np.mean(hists[k], axis=0), width=w,
                   color=CLASS_COLORS[k % 6], label=class_names[k])
    ax.set_xticks(np.arange(bins))
    ax.set(xlabel=f"uniform LBP code (0..{P} = number of 1-bits, {P + 1} = non-uniform)",
           ylabel="fraction of pixels", title="Mean LBP histogram per class")
    ax.legend(fontsize=7, ncol=3)
    return _save(fig, out_path)


def plot_glcm_distributions(F: np.ndarray, names: Sequence[str], y: np.ndarray,
                            class_names: Sequence[str], out_path, distance: int = 1) -> Path:
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    cols = [names.index(f"glcm_{p}_d{distance}_mean") for p in props if f"glcm_{p}_d{distance}_mean" in names]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, ci, p in zip(axes.ravel(), cols, props):
        data = [F[y == k, ci] for k in range(len(class_names))]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                        medianprops={"color": "black", "linewidth": 1.4})
        for patch, c in zip(bp["boxes"], CLASS_COLORS):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(class_names) + 1))
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=7)
        ax.set_title(f"GLCM {p} (d={distance}, mean over angles)", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("GLCM texture statistics per class", fontsize=10)
    return _save(fig, out_path)


def plot_depth_curve(depths: Sequence, train: Sequence[float], cv_mean: Sequence[float],
                     cv_std: Sequence[float], out_path, title: str) -> Path:
    x = np.arange(len(depths))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, train, "o-", label="training macro-F1")
    ax.errorbar(x, cv_mean, yerr=cv_std, fmt="s-", capsize=3, label="cross-validation macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set(xlabel="max_depth", ylabel="macro-F1", title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out_path)


def plot_importances(importances: np.ndarray, names: Sequence[str], out_path, title: str,
                     top: int = 20) -> Path:
    order = np.argsort(importances)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, 0.32 * top + 1.2))
    ax.barh(range(len(order))[::-1], importances[order], color="#4C72B0")
    ax.set_yticks(range(len(order))[::-1])
    ax.set_yticklabels([names[i] for i in order], fontsize=7)
    ax.set(xlabel="importance", title=title)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, out_path)


def plot_ablation(rows: List[Dict], group_key: str, out_path, title: str) -> Path:
    groups = list(dict.fromkeys(str(r[group_key]) for r in rows))
    series = list(dict.fromkeys(f"{r['feature_set'].upper()} + {r['model']}" for r in rows))
    x = np.arange(len(groups))
    w = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(groups) + 2) + (2.2 if len(series) > 1 else 0), 4.4))
    for i, s in enumerate(series):
        vals, errs = [], []
        for g in groups:
            r = [r for r in rows if str(r[group_key]) == g and f"{r['feature_set'].upper()} + {r['model']}" == s]
            vals.append(r[0]["cv_macro_f1_mean"] if r else np.nan)
            errs.append(r[0]["cv_macro_f1_std"] if r else 0)
        ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, yerr=errs, capsize=2, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=8)
    lo = np.nanmin([r["cv_macro_f1_mean"] - r["cv_macro_f1_std"] for r in rows])
    hi = np.nanmax([r["cv_macro_f1_mean"] + r["cv_macro_f1_std"] for r in rows])
    ax.set_ylim(max(0, lo - 0.05), max(1.0, hi) + 0.02)
    ax.set(ylabel="CV macro-F1 (training split)", title=title)
    ax.grid(axis="y", alpha=0.3)
    if len(series) > 1:
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="feature set + model",
                  title_fontsize=7)
    return _save(fig, out_path)
