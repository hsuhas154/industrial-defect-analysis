"""Whole-image similarity screen for re-processed copies, any image shape.

The NEU detector's second stage (background grain, ``duplicates.py``) was
calibrated on 200x200 steel images and compares non-square images on a
squashed square grid, so it is not used for KolektorSDD (500 x ~1260 px) or
DAGM. Exact shared content (identical files, identical pixels, overlapping
crops under any flip or rotation) is found for every dataset by the
shape-agnostic pixel-run search (E1 in ``duplicates.py``). This module covers
what E1 cannot: copies whose pixels were changed (re-compression, brightness or
contrast, blur, noise, a crop of a changed copy).

Method: every image is reduced by one fixed factor (so a crop keeps its scale)
and band-passed (difference of Gaussians, which removes illumination and keeps
texture), then standardized. For every pair, every allowed transform and every
shift with enough overlap, the masked normalized cross-correlation is computed
with FFTs on the GPU (Padfield 2010: all sums run over the overlap only). The
score of a pair is its maximum. ``copy_variants`` makes known re-processed
copies so the score a copy reaches is measured on the dataset itself, not
assumed; pairs between the copy scores and the bulk of distinct pairs go to a
person.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

DIHEDRAL = ["identity", "rot90", "rot180", "rot270",
            "flip_lr", "flip_lr+rot90", "flip_lr+rot180", "flip_lr+rot270"]
SHAPE_PRESERVING = (0, 2, 4, 6)   # identity, rot180, flip_lr, flip_ud (= flip_lr+rot180)


def band_thumb(img: np.ndarray, factor: float) -> np.ndarray:
    """Reduce by ``factor`` (INTER_AREA), band-pass, standardize."""
    h, w = img.shape
    th, tw = max(8, int(round(h / factor))), max(8, int(round(w / factor)))
    x = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
    fine = cv2.GaussianBlur(x, (0, 0), 0.6)
    band = fine - cv2.GaussianBlur(x, (0, 0), 4.0)
    sd = float(band.std())
    return (band - band.mean()) / sd if sd > 1e-3 else np.zeros_like(band)


def _dihedral(x: np.ndarray, k: int) -> np.ndarray:
    y = np.fliplr(x) if k >= 4 else x
    return np.ascontiguousarray(np.rot90(y, k % 4))


def _canvas_size(shapes: Sequence[Tuple[int, int]], transforms: Sequence[int]) -> Tuple[int, int]:
    """Padded FFT grid (P, Q): twice the largest image or transformed reference."""
    swap = any(k % 2 for k in transforms)
    H = max(max(h, w) if swap else h for h, w in shapes)
    W = max(max(h, w) if swap else w for h, w in shapes)
    return 2 * H, 2 * W


def gpu_bytes(shapes: Sequence[Tuple[int, int]], transforms: Sequence[int], n_refs: int | None = None,
              batch: int = 512) -> int:
    """GPU memory ``similarity_matrix`` needs: reference spectra plus one batch of work maps.

    The work term counts 24 float maps of the padded grid per batch item, more
    than the loop keeps alive at once.
    """
    P, Q = _canvas_size(shapes, transforms)
    refs = (len(shapes) if n_refs is None else n_refs) * len(transforms)
    return 3 * refs * P * (Q // 2 + 1) * 8 + min(batch, refs) * P * Q * 4 * 24


def similarity_matrix(thumbs: Sequence[np.ndarray], transforms: Sequence[int] = SHAPE_PRESERVING,
                      min_overlap: float = 0.5, device: str = "auto",
                      queries: Sequence[int] | None = None, batch: int = 512,
                      progress=None) -> Dict[str, np.ndarray]:
    """Best masked NCC over transforms and shifts for every pair of images.

    ``thumbs`` may differ in size. A shift counts only if the overlap covers at
    least ``min_overlap`` of the smaller image. Without ``queries`` all pairs
    are scored (each once, i < j) and ``score`` is the symmetric N x N matrix
    with -1 on the diagonal; with ``queries`` only those rows are computed
    against every other image. ``transform``, ``dy`` and ``dx`` give the best
    alignment of the second image of the pair: ``first[u + (dy, dx)]`` lies on
    ``transform(second)[u]`` on the reduced grid. In the all-pairs matrix the
    entries below the diagonal repeat those of the (i < j) computation.
    ``progress(done, total)`` is called after each query image.
    """
    import torch

    dev = torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
    N, T = len(thumbs), len(transforms)
    all_pairs = queries is None
    queries = list(range(N)) if all_pairs else list(queries)
    refs = [_dihedral(thumbs[j], k) for j in range(N) for k in transforms]   # j-major, k-minor
    H = max(max(t.shape[0] for t in thumbs), max(r.shape[0] for r in refs))
    W = max(max(t.shape[1] for t in thumbs), max(r.shape[1] for r in refs))
    P, Q = 2 * H, 2 * W
    rf = torch.fft.rfft2

    def canvas(arrs):
        x = np.zeros((len(arrs), P, Q), np.float32)
        m = np.zeros((len(arrs), P, Q), np.float32)
        for i, a in enumerate(arrs):
            x[i, :a.shape[0], :a.shape[1]] = a
            m[i, :a.shape[0], :a.shape[1]] = 1.0
        return torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev)

    def corr(FA, FB):   # sum_u A[u + d] B[u] for every shift d (circular on the padded grid)
        return torch.fft.irfft2(FA * torch.conj(FB), s=(P, Q))

    # Reference spectra once: 3 complex maps per (image, transform).
    if dev.type == "cuda":
        need = gpu_bytes([t.shape for t in thumbs], transforms, batch=batch)
        # free on the device plus what this process's allocator holds unused, minus a margin
        free = (torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()
                - torch.cuda.memory_allocated() - (256 << 20))
        if need > free:
            raise MemoryError(f"the screen needs {need / 1e9:.2f} GB on the GPU, "
                              f"{free / 1e9:.2f} GB is free: use a larger factor or free the GPU")
    FB, FB2, MB = [], [], []
    with torch.no_grad():
        for s0 in range(0, len(refs), batch):
            xb, mb = canvas(refs[s0:s0 + batch])
            FB.append(rf(xb)); FB2.append(rf(xb * xb)); MB.append(rf(mb))
        FB, FB2, MB = torch.cat(FB), torch.cat(FB2), torch.cat(MB)
    area = np.array([t.size for t in thumbs], dtype=np.float64)
    area_ref = np.repeat(area, T)
    dys = np.arange(P); dys = np.where(dys >= H, dys - P, dys)
    dxs = np.arange(Q); dxs = np.where(dxs >= W, dxs - Q, dxs)
    score = np.full((len(queries), N), -1.0, np.float32)
    best_t = np.zeros((len(queries), N), np.int8)
    best_dy = np.zeros((len(queries), N), np.int32)
    best_dx = np.zeros((len(queries), N), np.int32)
    tf = np.array(transforms)
    with torch.no_grad():
        for qi, i in enumerate(queries):
            r0 = (i + 1) * T if all_pairs else 0
            if r0 >= len(refs):
                if progress is not None:
                    progress(qi + 1, len(queries))
                continue
            xa, ma = canvas([thumbs[i]])
            Fa, Fa2, Ma = rf(xa), rf(xa * xa), rf(ma)
            vals, idxs = [], []
            for s0 in range(r0, len(refs), batch):
                s1 = min(s0 + batch, len(refs))
                Fb, Fb2, Mb = FB[s0:s1], FB2[s0:s1], MB[s0:s1]
                n = corr(Ma, Mb).clamp_min(1.0)
                sa, saa = corr(Fa, Mb), corr(Fa2, Mb)
                sb, sbb = corr(Ma, Fb), corr(Ma, Fb2)
                num = corr(Fa, Fb) - sa * sb / n
                den = torch.sqrt(((saa - sa * sa / n) * (sbb - sb * sb / n)).clamp_min(1e-6))
                need = torch.from_numpy(np.minimum(area_ref[s0:s1], area[i]) * min_overlap - 0.5) \
                    .to(dev, torch.float32)[:, None, None]
                ncc = torch.where(n >= need, num / den, torch.full_like(n, -2.0))
                v, ix = ncc.flatten(1).max(1)
                vals.append(v.cpu().numpy()); idxs.append(ix.cpu().numpy())
            v = np.concatenate(vals).reshape(-1, T)       # rows: images j from r0 // T on
            ix = np.concatenate(idxs).reshape(-1, T)
            js = np.arange(r0 // T, N)
            kbest = v.argmax(1)
            sel = v[np.arange(len(js)), kbest]
            u, w = np.divmod(ix[np.arange(len(js)), kbest], Q)
            keep = js != i
            js, kbest, sel, u, w = js[keep], kbest[keep], sel[keep], u[keep], w[keep]
            score[qi, js] = sel
            best_t[qi, js] = tf[kbest]
            best_dy[qi, js], best_dx[qi, js] = dys[u], dxs[w]
            if progress is not None:
                progress(qi + 1, len(queries))
    if all_pairs:
        lower = np.tril_indices(N, -1)
        score[lower] = score.T[lower]
        for a in (best_t, best_dy, best_dx):
            a[lower] = a.T[lower]
        np.fill_diagonal(score, -1.0)
    return {"score": score, "transform": best_t, "dy": best_dy, "dx": best_dx,
            "queries": np.array(queries)}


# --------------------------------------------------------------------------
# Planted copies: the scores a re-processed copy actually reaches
# --------------------------------------------------------------------------
def _jpeg(x: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def _clip(x) -> np.ndarray:
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def copy_variants(img: np.ndarray, rng: np.random.Generator, square: bool) -> Dict[str, np.ndarray]:
    """Re-processed copies of one image (each a plausible way a copy is made)."""
    h, w = img.shape
    f = img.astype(np.float32)
    out = {
        "jpeg_q50": _jpeg(img, 50),
        "jpeg_q75": _jpeg(img, 75),
        "brightness_contrast": _clip(0.8 * f + 30),
        "gamma_0.7": _clip(255.0 * (f / 255.0) ** 0.7),
        "blur_sigma1": cv2.GaussianBlur(img, (0, 0), 1.0),
        "noise_sigma4": _clip(f + rng.normal(0, 4, f.shape)),
        "crop_10pct": img[int(0.1 * h):, int(0.05 * w):],
        "crop_10pct+jpeg_q75": _jpeg(np.ascontiguousarray(img[:int(0.9 * h), int(0.05 * w):]), 75),
        "flip_lr+brightness": _clip(0.9 * np.fliplr(f) + 10),
    }
    if square:
        out["rot90+jpeg_q75"] = _jpeg(np.ascontiguousarray(np.rot90(img)), 75)
    return out


def pair_score(a: np.ndarray, b: np.ndarray, factor: float, transforms: Sequence[int],
               min_overlap: float = 0.5, device: str = "auto") -> Tuple[float, int, int, int]:
    """(score, transform, dy, dx) of one pair at reduction ``factor``."""
    r = similarity_matrix([band_thumb(a, factor), band_thumb(b, factor)], transforms,
                          min_overlap, device, queries=[0])
    return (float(r["score"][0, 1]), int(r["transform"][0, 1]), int(r["dy"][0, 1]), int(r["dx"][0, 1]))
