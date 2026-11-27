"""Figures for the Phase 4 datasets, all at native resolution (one image pixel
is at least one figure pixel).

* KolektorSDD samples with their masks.
* The two byte-identical part copies: one defect annotated twice.
* The kos38 / kos39 re-capture: the same commutator imaged twice.
* DAGM: a normal and a defective image per class with the weak ellipse label.
* DAGM: the most similar pair judged distinct in each class, for a visual check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

from idefect.data.dagm import load_dagm, read_label
from idefect.data.duplicates import overlap_views
from idefect.data.kolektor import load_kolektorsdd, read_image, read_mask
from idefect.data.similarity_screen import DIHEDRAL, _dihedral
from idefect.visualization.plots import _ensure_parent, plt

DPI = 200


def _contour(ax, mask: np.ndarray, color: str, lw: float = 0.8) -> None:
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw)


def _show(ax, img, title: str = "", fs: int = 8) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(title, fontsize=fs)
    ax.set_xticks([])
    ax.set_yticks([])


def _refine(a: np.ndarray, b: np.ndarray, dy: int, dx: int, r: int = 2):
    """Shift within +-r px of (dy, dx) with the smallest mean absolute difference
    (the screen's shift is measured on reduced images)."""
    best = None
    for ddy in range(-r, r + 1):
        for ddx in range(-r, r + 1):
            va, vb = overlap_views(a, b, dy + ddy, dx + ddx)
            if va.size == 0:
                continue
            d = float(np.abs(va - vb).mean())
            if best is None or d < best[0]:
                best = (d, dy + ddy, dx + ddx)
    return best[1], best[2]


def _panel_inches(px: int) -> float:
    return px / DPI * 1.05


def kolektor_samples(samples, out: Path, n: int = 6) -> Path:
    defect = [s for s in samples if s.defective][::max(1, sum(s.defective for s in samples) // n)][:n]
    normal = [s for s in samples if not s.defective][::40][:n]
    w = _panel_inches(500)
    fig, axes = plt.subplots(2, n, figsize=(w * n, w * 2.6 * 2 + 0.6), layout="constrained")
    for ax, s in zip(axes[0], defect):
        _show(ax, read_image(s.image), f"{s.id} (defective, {s.defect_px} px)")
        _contour(ax, read_mask(s.mask), "red")
    for ax, s in zip(axes[1], normal):
        _show(ax, read_image(s.image), f"{s.id} (normal)")
    fig.suptitle("KolektorSDD: defective images (red: annotated defect) and normal images", fontsize=11)
    fig.savefig(_ensure_parent(out), dpi=DPI)
    plt.close(fig)
    return out


def _crop_box(mask: np.ndarray, size: int = 256):
    ys, xs = np.nonzero(mask)
    cy, cx = int(ys.mean()), int(xs.mean())
    h, w = mask.shape
    y0 = int(np.clip(cy - size // 2, 0, h - size))
    x0 = int(np.clip(cx - size // 2, 0, w - size))
    return y0, x0, size


def kolektor_copies(samples, pairs: Sequence[Dict], out: Path) -> Optional[Path]:
    """One row per twin defect: both annotations of the same pixels."""
    if not pairs:
        return None
    by = {s.id: s for s in samples}
    w = _panel_inches(256) * 1.8
    fig, axes = plt.subplots(len(pairs), 4, figsize=(w * 4, w * len(pairs) + 0.7), squeeze=False,
                             layout="constrained")
    for row, p in zip(axes, pairs):
        a, b = by[p["a"]], by[p["b"]]
        img = read_image(a.image)
        ma, mb = read_mask(a.mask), read_mask(b.mask)
        y0, x0, sz = _crop_box(ma | mb)
        crop = (slice(y0, y0 + sz), slice(x0, x0 + sz))
        _show(row[0], img[crop], f"{a.id} = {b.id}\n(byte-identical image)")
        _show(row[1], img[crop], f"mask of {a.id}\n({a.defect_px} px)")
        _contour(row[1], ma[crop], "red")
        _show(row[2], img[crop], f"mask of {b.id}\n({b.defect_px} px)")
        _contour(row[2], mb[crop], "lime")
        rgb = np.zeros((sz, sz, 3), np.uint8)
        rgb[..., 0] = ma[crop] * 255
        rgb[..., 1] = mb[crop] * 255
        row[3].imshow(rgb, interpolation="nearest")
        row[3].set_title(f"red: {a.part} only, green: {b.part} only\nyellow: both, IoU {p['mask_iou']:.3f}",
                         fontsize=8)
        row[3].set_xticks([])
        row[3].set_yticks([])
    fig.suptitle("KolektorSDD byte-identical parts: the same defect, annotated twice (256 x 256 px crops)",
                 fontsize=11)
    fig.savefig(_ensure_parent(out), dpi=DPI)
    plt.close(fig)
    return out


def kolektor_recapture(samples, pairs: Sequence[Dict], related_parts: Sequence[str], out: Path,
                       out_zoom: Path, fine_factor: float, best_distinct: Optional[float]):
    """Matched images of parts related by content (not byte copies), and a zoom."""
    rel = set(related_parts)
    pairs = [p for p in pairs if p["a"].split("/")[0] != p["b"].split("/")[0]
             and p["a"].split("/")[0] in rel and p["b"].split("/")[0] in rel]
    if not pairs:
        return None, None
    by = {s.id: s for s in samples}
    pairs = sorted(pairs, key=lambda p: by[p["a"]].index)
    n = len(pairs)
    w = _panel_inches(500)
    fig, axes = plt.subplots(2, n, figsize=(w * n, w * 2.6 * 2 + 0.8), squeeze=False, layout="constrained")
    for k, p in enumerate(pairs):
        a, b = by[p["a"]], by[p["b"]]
        _show(axes[0, k], read_image(a.image), a.id)
        _contour(axes[0, k], read_mask(a.mask), "red")
        _show(axes[1, k], read_image(b.image), f"{b.id}\ncoarse {p['coarse']:.3f}, fine {p['fine']:.3f}")
        _contour(axes[1, k], read_mask(b.mask), "lime")
    parts = sorted({p["a"].split("/")[0] for p in pairs} | {p["b"].split("/")[0] for p in pairs})
    other = "" if best_distinct is None else f"; the best pair judged distinct scores {best_distinct:.2f}"
    fig.suptitle(f"KolektorSDD {' and '.join(parts)}: every image has one matching partner{other}",
                 fontsize=11)
    fig.savefig(_ensure_parent(out), dpi=DPI)
    plt.close(fig)

    zp = [p for p in pairs if by[p["a"]].defective or by[p["b"]].defective]
    zp = zp or pairs[:1]
    w = _panel_inches(256) * 1.8
    fig, axes = plt.subplots(len(zp), 3, figsize=(w * 3, w * len(zp) + 0.7), squeeze=False,
                             layout="constrained")
    for row, p in zip(axes, zp):
        a, b = by[p["a"]], by[p["b"]]
        ia, ib = read_image(a.image).astype(np.float32), read_image(b.image).astype(np.float32)
        t = DIHEDRAL.index(p["transform"])
        ib_t = _dihedral(ib, t)
        dy, dx = int(round(-p["shift"][0] * fine_factor)), int(round(-p["shift"][1] * fine_factor))
        dy, dx = _refine(ia, ib_t, dy, dx, int(np.ceil(fine_factor)))
        va, vb = overlap_views(ia, ib_t, dy, dx)
        ma = read_mask(a.mask)
        mb = _dihedral(read_mask(b.mask), t)
        mav, mbv = overlap_views(ma.astype(np.uint8), mb.astype(np.uint8), dy, dx)
        box_mask = (mav | mbv) if (mav.any() or mbv.any()) else np.ones_like(mav)
        y0, x0, sz = _crop_box(box_mask)
        c = (slice(y0, y0 + sz), slice(x0, x0 + sz))
        _show(row[0], va[c], f"{a.id}")
        _contour(row[0], mav[c] > 0, "red")
        _show(row[1], vb[c], f"{b.id} (aligned)")
        _contour(row[1], mbv[c] > 0, "lime")
        row[2].imshow(np.abs(va[c] - vb[c]), cmap="magma", vmin=0, vmax=60, interpolation="nearest")
        within = float((np.abs(va - vb) <= 2).mean())
        row[2].set_title(f"|difference| (whole image: mean {np.abs(va - vb).mean():.1f},\n"
                         f"{100 * within:.0f}% of pixels within 2 grey levels):\n"
                         "a second capture, not a file copy", fontsize=8)
        row[2].set_xticks([])
        row[2].set_yticks([])
    fig.suptitle("Re-captured part: the same defect in both parts (256 x 256 px crops)", fontsize=11)
    fig.savefig(_ensure_parent(out_zoom), dpi=DPI)
    plt.close(fig)
    return out, out_zoom


def dagm_samples(samples, out: Path, classes: Sequence[int]) -> Path:
    w = _panel_inches(512)
    fig, axes = plt.subplots(3, len(classes), figsize=(w * len(classes), w * 3 + 0.8), squeeze=False,
                             layout="constrained")
    for k, c in enumerate(classes):
        normal = next(s for s in samples if s.cls == c and s.split == "Train" and not s.defective)
        defect = next(s for s in samples if s.cls == c and s.split == "Train" and s.defective)
        _show(axes[0, k], read_image(normal.image), f"Class{c} normal\n{normal.split}/{normal.name}")
        img = read_image(defect.image)
        _show(axes[1, k], img, f"Class{c} defective\n{defect.split}/{defect.name}")
        _show(axes[2, k], img, "weak label (ellipse)")
        _contour(axes[2, k], read_label(defect.label, img.shape), "red", 1.0)
    fig.suptitle("DAGM 2007: one normal and one defective image per class; the label is a coarse ellipse "
                 "around the defect, not a pixel mask", fontsize=12)
    fig.savefig(_ensure_parent(out), dpi=DPI)
    plt.close(fig)
    return out


def _aligned(by: Dict, p: Dict, fine_factor: float):
    """Overlap views of a scored pair at native resolution, B transformed and shifted onto A."""
    a = read_image(by[p["a"]].image).astype(np.float32)
    b = _dihedral(read_image(by[p["b"]].image).astype(np.float32), DIHEDRAL.index(p["transform"]))
    dy, dx = int(round(-p["shift"][0] * fine_factor)), int(round(-p["shift"][1] * fine_factor))
    dy, dx = _refine(a, b, dy, dx, int(np.ceil(fine_factor)))
    return overlap_views(a, b, dy, dx)


def dagm_review_pairs(samples, per_class: Dict, out: Path, fine_factor: float, review: float,
                      same: float) -> Optional[Path]:
    """Every pair in the review band, with the person's decision, plus each class's best pair below it."""
    rows = []
    for c, pc in sorted(per_class.items(), key=lambda kv: int(kv[0])):
        pairs = [(p, f"decided {p['decision']}") for p in pc.get("review_decided", [])]
        pairs += [(p, "AWAITING REVIEW") for p in pc.get("review", [])]
        if not pairs:
            continue
        rows += sorted(pairs, key=lambda x: -x[0]["fine"])
        if pc["content"]["top_distinct"]:
            rows.append((pc["content"]["top_distinct"][0], f"best pair below {review:.2f}"))
    if not rows:
        return None
    by = {s.id: s for s in samples}
    w = _panel_inches(512)
    fig, axes = plt.subplots(len(rows), 4, figsize=(w * 4, w * len(rows) + 1.0),
                             squeeze=False, layout="constrained")
    for row, (p, tag) in zip(axes, rows):
        va, vb = _aligned(by, p, fine_factor)
        y0, x0 = max(0, va.shape[0] // 2 - 64), max(0, va.shape[1] // 2 - 64)
        name = {s: ("DEFECTIVE" if by[s].defective else "normal") for s in (p["a"], p["b"])}
        _show(row[0], va, f"{tag}\nfine {p['fine']:.4f}, coarse {p['coarse']:.4f}\n{p['a']} ({name[p['a']]})")
        _show(row[1], vb, f"B: {p['transform']}, shift {p['shift'][0]},{p['shift'][1]}\n(reduced grid)\n"
                          f"{p['b']} ({name[p['b']]})")
        _show(row[2], va[y0:y0 + 128, x0:x0 + 128], "A, centre 128 x 128 px")
        _show(row[3], vb[y0:y0 + 128, x0:x0 + 128], "B, same place")
    fig.suptitle(f"DAGM pairs in the review band (fine score {review:.2f} to {same:.2f}) with the decision "
                 f"recorded in the review file,\nand the best pair below the band in the same class",
                 fontsize=11)
    fig.savefig(_ensure_parent(out), dpi=DPI)
    plt.close(fig)
    return out


def dagm_closest_distinct(samples, per_class: Dict, out: Path, fine_factor: float,
                          same: float) -> Optional[Path]:
    rows = []
    for c, pc in sorted(per_class.items(), key=lambda kv: int(kv[0])):
        td = pc["content"]["top_distinct"]
        if td:
            rows.append((int(c), td[0]))
    if not rows:
        return None
    by = {s.id: s for s in samples}
    planted_min = min(v["fine_min"] for pc in per_class.values() for v in pc["content"]["planted"].values())
    w = _panel_inches(512)
    fig, axes = plt.subplots(len(rows), 3, figsize=(w * 3, w * len(rows) + 0.8),
                             squeeze=False, layout="constrained")
    for row, (c, p) in zip(axes, rows):
        va, vb = _aligned(by, p, fine_factor)
        _show(row[0], va, f"{p['a']} (overlap)")
        _show(row[1], vb, f"{p['b']} [{p['transform']}]")
        row[2].imshow(np.abs(va - vb), cmap="magma", vmin=0, vmax=60, interpolation="nearest")
        row[2].set_title(f"|A - B|   coarse {p['coarse']:.3f}, fine {p['fine']:.3f}", fontsize=8)
        row[2].set_xticks([])
        row[2].set_yticks([])
    fig.suptitle(f"DAGM: the most similar pair below the review band in each class\n(fine score decides: "
                 f"same at {same:.2f} or more; planted copies scored {planted_min:.3f} or more)", fontsize=11)
    fig.savefig(_ensure_parent(out), dpi=DPI)
    plt.close(fig)
    return out


def make_phase4_figures(cfg, report: Dict) -> Dict[str, Path]:
    fig_dir = Path(cfg.output.figures_dir)
    out: Dict[str, Path] = {}
    k = report.get("kolektorsdd")
    if k:
        samples = load_kolektorsdd(cfg.kolektorsdd.root)
        out["kolektorsdd_samples"] = kolektor_samples(samples, fig_dir / "phase4_kolektorsdd_samples.png")
        p = kolektor_copies(samples, k["copy_mask_agreement"], fig_dir / "phase4_kolektorsdd_twin_masks.png")
        if p:
            out["kolektorsdd_twin_masks"] = p
        related = [p for r in k["splits"]["relations"] if r["kind"] == "related" for p in r["parts"]]
        a, b = kolektor_recapture(samples, k["cross_part_pairs"], related,
                                  fig_dir / "phase4_kolektorsdd_recapture.png",
                                  fig_dir / "phase4_kolektorsdd_recapture_zoom.png",
                                  cfg.kolektorsdd.screen.fine_factor, k["content"]["max_fine_distinct"])
        if a:
            out["kolektorsdd_recapture"], out["kolektorsdd_recapture_zoom"] = a, b
    d = report.get("dagm")
    if d:
        samples = load_dagm(cfg.dagm.root, cfg.dagm.classes)
        out["dagm_samples"] = dagm_samples(samples, fig_dir / "phase4_dagm_samples.png", cfg.dagm.classes)
        p = dagm_closest_distinct(samples, d["per_class"], fig_dir / "phase4_dagm_closest_distinct.png",
                                  cfg.dagm.screen.fine_factor, cfg.dagm.screen.same)
        if p:
            out["dagm_closest_distinct"] = p
        p = dagm_review_pairs(samples, d["per_class"], fig_dir / "phase4_dagm_review_pairs.png",
                              cfg.dagm.screen.fine_factor, cfg.dagm.screen.review, cfg.dagm.screen.same)
        if p:
            out["dagm_review_pairs"] = p
    return out
