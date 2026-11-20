"""Figures for checking near-duplicate decisions by eye.

Every row shows one pair at the alignment the detector found: the overlap of
image A, the same region of image B (after its flip/rotation), and the absolute
difference. Copies show a dark difference map apart from re-processing noise;
distinct images show texture everywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from idefect.data.duplicates import DIHEDRAL_NAMES, PairResult, dihedral, overlap_views
from idefect.visualization.plots import _ensure_parent, plt, tidy


def plot_pairs(pairs: Sequence[PairResult], out_path: str | Path, title: str,
               max_rows: int = 12) -> Path | None:
    pairs = [p for p in pairs if p.transform in DIHEDRAL_NAMES][:max_rows]
    if not pairs:
        return None
    # Constrained layout: row heights fit the multi-line captions and the
    # figure title gets its own space above the first row.
    fig, axes = plt.subplots(len(pairs), 3, figsize=(8.4, 3.0 * len(pairs) + 0.5), squeeze=False,
                             layout="constrained")
    for row, p in zip(axes, pairs):
        a = cv2.imread(p.a, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        b = dihedral(cv2.imread(p.b, cv2.IMREAD_GRAYSCALE), DIHEDRAL_NAMES.index(p.transform))
        va, vb = overlap_views(a, b.astype(np.float32), p.shift[0], p.shift[1])
        row[0].imshow(va, cmap="gray", vmin=0, vmax=255)
        row[1].imshow(vb, cmap="gray", vmin=0, vmax=255)
        row[2].imshow(np.abs(va - vb), cmap="magma", vmin=0, vmax=40)
        row[0].set_title(f"{Path(p.a).name} (overlap)", fontsize=8)
        row[1].set_title(f"{Path(p.b).name} [{p.transform}, shift {p.shift[0]},{p.shift[1]}]", fontsize=8)
        grain = "n/a" if p.grain is None else f"{p.grain:.2f}"
        row[2].set_title(f"|A-B|  {p.kind} ({p.evidence})\npixels within 2: {p.exact_fraction:.2f}, "
                         f"grain {grain} ({p.grain_tiles} tiles)", fontsize=7)
        for ax in row:
            ax.axis("off")
    fig.suptitle(title, fontsize=10)
    out_path = _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_group_sizes(groups: Sequence[Sequence[str]], out_path: str | Path) -> Path:
    sizes = [len(g) for g in groups]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    if sizes:
        bins = np.arange(2, max(sizes) + 2) - 0.5
        ax.hist(sizes, bins=bins, color="#4C72B0")
    ax.set(xlabel="images per group", ylabel="groups",
           title=f"Near-duplicate groups ({len(groups)} groups, {sum(sizes)} images)")
    ax.grid(axis="y", alpha=0.3)
    tidy(fig)
    out_path = _ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
