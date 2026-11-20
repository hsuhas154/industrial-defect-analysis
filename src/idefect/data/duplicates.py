"""Exact and near-duplicate detection with explicit, calibrated verification.

Leakage happens when the same visual content appears on both sides of a split.
That content can be an identical file, a flipped or rotated copy, an overlapping
crop of the same capture, or a re-processed copy (brightness, contrast,
resampling, mild compression). A detector must group all of those and must
never group two genuinely different images. Analysis of NEU-CLS showed why
simple similarity scores cannot do this: different crops along the *same long
scratch* reach a raw correlation of 0.998 (the line dominates), while a
JPEG-30 copy of an image can drop to 0.86. So the decision uses two pieces of
evidence that a different image cannot fake:

**E1, pixel identity.** At the best alignment (any of the 8 flips/rotations and
any shift), at least ``exact_fraction`` of the overlapping pixels agree within
``exact_tolerance`` grey levels, over at least ``min_exact_overlap_px`` pixels.
Real overlaps in NEU agree on 100 percent of pixels; all other pairs never
exceed 74 percent. Exact overlaps are found exhaustively by hashing short horizontal
pixel runs of every image and of every flipped/rotated version: any shared
textured region a few pixels high produces matching runs that vote for one
alignment, so no exact overlap can be missed.

**E2, shared background grain.** For re-processed copies, the fine-scale grain
of the steel surface away from strong structures (lines, edges, blobs) is the
fingerprint: a copy shares it, a different crop of the same defect does not.
The overlap is band-passed (difference of Gaussians), strong structures and
saturated pixels are masked out with a wide margin, and the normalized
cross-correlation is computed per 16x16 tile that is mostly background. The
score is the median over qualifying tiles, and at least ``grain_min_tiles``
tiles are required for the score to count as evidence. On NEU, pairs from
different classes that share no pixels never exceed 0.32 and no pair judged
distinct exceeds 0.385, while pixel-identical pairs (copies by construction)
score 1.0 and planted re-processed copies down to JPEG quality 50 score 0.54
or more.

Decision for a pair: **same** if E1 holds or the grain score is at least
``grain_same``; **review** if the grain score is in
[``grain_review``, ``grain_same``); otherwise **distinct**. Pairs whose overlap has
too little background for E2 and that fail E1 are kept apart but recorded as
**unverified**, so a person can inspect them. Review decisions live in a YAML
file (``same`` / ``distinct`` lists) and override the detector. Unresolved
review pairs are grouped, the safe choice for leakage, and reported.

Groups are connected components of the "same" relation (union-find), so if
A~B and B~C, then A, B and C stay together.

``match_against`` checks one external image (for example an upload in the
demo) against the dataset: whole-image copies at any resolution through the
correlation search, and crops at their original scale through the same
pixel-run voting as E1.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from idefect.config import DuplicateConfig

DETECTOR_VERSION = "3"

DIHEDRAL_NAMES = [
    "identity", "rot90", "rot180", "rot270",
    "flip_lr", "flip_lr+rot90", "flip_lr+rot180", "flip_lr+rot270",
]


@dataclass
class PairResult:
    a: str
    b: str
    kind: str              # same | review | distinct | unverified
    evidence: str          # identical-file | pixel-identical | grain | none
    exact_fraction: float  # fraction of overlap pixels within tolerance
    grain: Optional[float]
    grain_tiles: int
    overlap_px: int
    transform: str
    shift: Tuple[int, int]
    decided_by: str = "detector"   # detector | review:same | review:distinct


@dataclass
class DuplicateReport:
    groups: List[List[str]]
    pairs: List[PairResult]
    n_images: int
    n_candidates: int
    config: Dict = field(default_factory=dict)

    def _sel(self, kind: str) -> List[PairResult]:
        return [p for p in self.pairs if p.kind == kind and p.decided_by == "detector"]

    @property
    def unresolved_review(self) -> List[PairResult]:
        return self._sel("review")

    @property
    def unverified(self) -> List[PairResult]:
        return self._sel("unverified")

    def max_distinct_grain(self) -> Optional[float]:
        vals = [p.grain for p in self.pairs if p.kind == "distinct" and p.grain is not None]
        return max(vals) if vals else None

    def summary(self) -> Dict:
        kinds: Dict[str, int] = {}
        for p in self.pairs:
            key = p.kind if p.decided_by == "detector" else p.decided_by
            kinds[key] = kinds.get(key, 0) + 1
        evidence: Dict[str, int] = {}
        for p in self.pairs:
            if p.kind == "same":
                evidence[p.evidence] = evidence.get(p.evidence, 0) + 1
        return {
            "n_images": self.n_images,
            "n_candidates_verified": self.n_candidates,
            "n_groups": len(self.groups),
            "n_images_in_groups": sum(len(g) for g in self.groups),
            "largest_group": max((len(g) for g in self.groups), default=0),
            "pairs_by_decision": kinds,
            "same_pairs_by_evidence": evidence,
            "max_grain_among_distinct": self.max_distinct_grain(),
        }

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["summary"] = self.summary()
        return d


class DuplicateCacheMissing(RuntimeError):
    """Raised by ``detect_duplicates(..., cache_only=True)`` when no cached result
    matches, so callers such as the demo never start the full detection."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def pixel_hash(img: np.ndarray) -> str:
    h = hashlib.md5()
    h.update(str(img.shape).encode())
    h.update(np.ascontiguousarray(img).tobytes())
    return h.hexdigest()


def _read_gray(path: str) -> Optional[np.ndarray]:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def dihedral(x: np.ndarray, k: int) -> np.ndarray:
    """The k-th element of the dihedral group (flips and 90-degree rotations)."""
    y = np.fliplr(x) if k >= 4 else x
    return np.ascontiguousarray(np.rot90(y, k % 4))


def overlap_views(a: np.ndarray, b: np.ndarray, dy: int, dx: int):
    """Overlapping parts of ``a`` and ``b`` when ``b`` is displaced so that
    ``a[u]`` lies on ``b[u + (dy, dx)]``. Works for any two image shapes."""
    ha, wa = a.shape
    hb, wb = b.shape
    y0, y1 = max(0, -dy), min(ha, hb - dy)
    x0, x1 = max(0, -dx), min(wa, wb - dx)
    if y1 <= y0 or x1 <= x0:
        return a[:0, :0], b[:0, :0]
    return a[y0:y1, x0:x1], b[y0 + dy:y1 + dy, x0 + dx:x1 + dx]


def texture_image(img: np.ndarray, size: int) -> Tuple[np.ndarray, bool]:
    """Band-pass, standardized image at ``size`` x ``size`` for alignment search.

    Returns ``(texture, featureless)``; featureless images can only be matched
    by E1.
    """
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    s_small = 0.75 * size / 192.0
    s_large = 8.0 * size / 192.0
    fine = cv2.GaussianBlur(x, (0, 0), s_small) if s_small >= 0.3 else x
    band = fine - cv2.GaussianBlur(x, (0, 0), s_large)
    sd = float(band.std())
    if sd < 0.25:
        return np.zeros_like(band), True
    return (band - band.mean()) / sd, False


# --------------------------------------------------------------------------
# E1 and E2 measurements at a given alignment
# --------------------------------------------------------------------------
def exact_agreement(va: np.ndarray, vb: np.ndarray, tol: int) -> float:
    if va.size == 0:
        return 0.0
    return float((np.abs(va.astype(np.int16) - vb.astype(np.int16)) <= tol).mean())


def _structure_strength(v: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(v.astype(np.float32), (0, 0), 2.0)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def background_mask(va: np.ndarray, vb: np.ndarray, grow: int) -> np.ndarray:
    """Pixels that are plain background in both images: strong structures
    (robust outliers of gradient strength) and saturated pixels are removed,
    with a ``grow``-pixel margin so line halos are excluded too."""
    s = np.maximum(_structure_strength(va), _structure_strength(vb))
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med)) + 1e-6
    bad = (s > med + 4 * mad) | (va >= 250) | (vb >= 250) | (va <= 5) | (vb <= 5)
    bad = cv2.dilate(bad.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
    m = ~bad
    m[:4] = m[-4:] = False
    m[:, :4] = m[:, -4:] = False
    return m


def grain_score(va: np.ndarray, vb: np.ndarray, cfg: DuplicateConfig) -> Tuple[Optional[float], int]:
    """Median per-tile NCC of the fine band over background tiles.

    Returns ``(score, n_tiles)``; ``score`` is None when fewer than
    ``grain_min_tiles`` tiles qualify (not enough evidence either way).
    """
    if va.shape[0] < cfg.grain_tile or va.shape[1] < cfg.grain_tile:
        return None, 0
    fa, fb = va.astype(np.float32), vb.astype(np.float32)
    ba = cv2.GaussianBlur(fa, (0, 0), 1.0) - cv2.GaussianBlur(fa, (0, 0), 2.5)
    bb = cv2.GaussianBlur(fb, (0, 0), 1.0) - cv2.GaussianBlur(fb, (0, 0), 2.5)
    m = background_mask(fa, fb, cfg.grain_mask_grow)
    t = cfg.grain_tile
    vals = []
    for y in range(0, va.shape[0] - t + 1, t):
        for x in range(0, va.shape[1] - t + 1, t):
            mm = m[y:y + t, x:x + t]
            if mm.mean() < 0.7:
                continue
            p = ba[y:y + t, x:x + t][mm]
            q = bb[y:y + t, x:x + t][mm]
            p = p - p.mean()
            q = q - q.mean()
            den = float(np.sqrt((p * p).sum() * (q * q).sum()))
            if den > 1e-6:
                vals.append(float((p * q).sum() / den))
    if len(vals) < cfg.grain_min_tiles:
        return None, len(vals)
    return float(np.median(vals)), len(vals)


def evaluate_alignment(a: np.ndarray, b: np.ndarray, k: int, dy: int, dx: int,
                       cfg: DuplicateConfig) -> Dict:
    """E1 and E2 for ``b`` transformed by dihedral element ``k`` and shifted."""
    bt = dihedral(b, k)
    va, vb = overlap_views(a, bt, dy, dx)
    frac = exact_agreement(va, vb, cfg.exact_tolerance)
    exact = frac >= cfg.exact_fraction and va.size >= cfg.min_exact_overlap_px
    min_px = cfg.min_overlap * min(a.size, b.size)
    grain, tiles = grain_score(va, vb, cfg) if va.size >= min_px else (None, 0)
    return {"k": k, "dy": dy, "dx": dx, "exact_fraction": frac, "exact": exact,
            "grain": grain, "tiles": tiles, "overlap_px": int(va.size)}


def decide(ev: Dict, cfg: DuplicateConfig) -> Tuple[str, str]:
    """Map E1/E2 measurements to (kind, evidence)."""
    if ev["exact"]:
        return "same", "pixel-identical"
    g = ev["grain"]
    if g is None:
        return "unverified", "none"
    if g >= cfg.grain_same:
        return "same", "grain"
    if g >= cfg.grain_review:
        return "review", "grain"
    return "distinct", "grain"


_RANK = {"same": 3, "review": 2, "unverified": 1, "distinct": 0}


def _better(ev1: Dict, ev2: Optional[Dict], cfg: DuplicateConfig) -> bool:
    if ev2 is None:
        return True
    k1, k2 = _RANK[decide(ev1, cfg)[0]], _RANK[decide(ev2, cfg)[0]]
    if k1 != k2:
        return k1 > k2
    return (ev1["exact_fraction"], ev1["grain"] or -2) > (ev2["exact_fraction"], ev2["grain"] or -2)


# --------------------------------------------------------------------------
# E1 candidate search: hashed pixel runs with alignment voting
# --------------------------------------------------------------------------
def _run_hashes(img: np.ndarray, L: int, row_step: int, col_step: int, min_range: int, R: np.ndarray):
    """Hashes of horizontal runs of length L (rows/cols subsampled)."""
    win = np.lib.stride_tricks.sliding_window_view(img, L, axis=1)[::row_step, ::col_step]
    rng = win.max(-1).astype(np.int16) - win.min(-1).astype(np.int16)
    keep = rng >= min_range
    h = (win.astype(np.uint64) * R).sum(-1, dtype=np.uint64)
    rows, cols = np.nonzero(keep)
    return h[keep], rows * row_step, cols * col_step


def exact_overlap_candidates(images: List[np.ndarray], idx: List[int], cfg: DuplicateConfig):
    """(i, j) -> (k, dy, dx) alignments proposed by matching pixel runs."""
    L, s = cfg.segment_len, cfg.segment_stride
    R = np.random.default_rng(12345).integers(1, 2**63 - 1, size=L, dtype=np.uint64) | np.uint64(1)
    keys, meta = [], []
    for j in idx:
        for k in range(8):
            h, r, c = _run_hashes(dihedral(images[j], k), L, s, s, cfg.min_segment_range, R)
            keys.append(h)
            meta.append(np.stack([np.full_like(r, j), np.full_like(r, k), r, c], 1).astype(np.int32))
    if not keys:
        return {}
    keys = np.concatenate(keys)
    meta = np.concatenate(meta)
    order = np.argsort(keys, kind="stable")
    keys, meta = keys[order], meta[order]
    uniq, start, counts = np.unique(keys, return_index=True, return_counts=True)
    ok = counts <= cfg.max_key_occurrences
    uniq, start, counts = uniq[ok], start[ok], counts[ok]
    if len(uniq) == 0:
        return {}

    votes: Dict[Tuple[int, int, int, int, int], int] = {}
    for i in idx:
        h, r, c = _run_hashes(images[i], L, 1, 1, cfg.min_segment_range, R)
        pos = np.clip(np.searchsorted(uniq, h), 0, len(uniq) - 1)
        for t in np.nonzero(uniq[pos] == h)[0]:
            p = pos[t]
            for e in range(start[p], start[p] + counts[p]):
                j, k, rb, cb = meta[e]
                if j == i:
                    continue
                key = (i, int(j), int(k), int(rb - r[t]), int(cb - c[t]))
                votes[key] = votes.get(key, 0) + 1
    best: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}
    for (i, j, k, dy, dx), v in votes.items():
        if v >= cfg.min_votes and ((i, j) not in best or v > best[(i, j)][0]):
            best[(i, j)] = (v, k, dy, dx)
    return {pair: (k, dy, dx) for pair, (v, k, dy, dx) in best.items()}


# --------------------------------------------------------------------------
# E2 candidate search: GPU masked NCC over all pairs, shifts and transforms
# --------------------------------------------------------------------------
def _torch_device(pref: str):
    import torch

    if pref == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _dihedral_t(x, k: int):
    import torch

    y = torch.flip(x, dims=(-1,)) if k >= 4 else x
    return torch.rot90(y, k % 4, dims=(-2, -1))


class _NCCEngine:
    """Masked NCC maps between SxS images for every shift at once (FFT based).

    NCC(s) = (Sab - Sa Sb / n) / sqrt((Saa - Sa^2/n)(Sbb - Sb^2/n)), all sums over
    the overlap (Padfield 2010). Only Sab depends on the pair.
    """

    def __init__(self, size: int, min_overlap: float, device):
        import torch

        self.torch = torch
        self.S, self.P, self.device = size, 2 * size, device
        ones = torch.zeros((self.P, self.P), device=device)
        ones[:size, :size] = 1.0
        self.M = torch.fft.rfft2(ones)
        self.n = self._corr(self.M, self.M).clamp_min(1.0)
        self.valid = self.n >= min_overlap * size * size - 0.5
        self.weight = torch.sqrt(self.n / (size * size))

    def _corr(self, FX, FY):
        return self.torch.fft.irfft2(self.torch.conj(FX) * FY, s=(self.P, self.P))

    def spectra(self, x):
        xp = self.torch.zeros(x.shape[:-2] + (self.P, self.P), device=self.device)
        xp[..., : self.S, : self.S] = x
        return self.torch.fft.rfft2(xp), self.torch.fft.rfft2(xp * xp)

    def a_terms(self, F, Fsq):
        return self._corr(F, self.M.expand_as(F)), self._corr(Fsq, self.M.expand_as(Fsq))

    def b_terms(self, F, Fsq):
        Mx = self.M.expand_as(F)
        return self._corr(Mx, F), self._corr(Mx, Fsq)

    def ncc(self, Fa, Sa, Saa, Fb, Sb, Sbb):
        torch = self.torch
        n = self.n
        num = self._corr(Fa, Fb) - Sa * Sb / n
        den = torch.sqrt((Saa - Sa * Sa / n).clamp_min(0) * (Sbb - Sb * Sb / n).clamp_min(0))
        out = torch.where(den > 1e-3 * n, num / den.clamp_min(1e-12), torch.full_like(num, -1.0))
        return torch.where(self.valid, out, torch.full_like(out, -1.0))

    def index_to_shift(self, idx: int) -> Tuple[int, int]:
        r, c = divmod(int(idx), self.P)
        return (r if r < self.S else r - self.P), (c if c < self.S else c - self.P)


def ncc_candidates(images, idx: List[int], cfg: DuplicateConfig,
                   query_only: bool = False) -> List[Tuple[int, int]]:
    """Each image's ``top_k`` most correlated partners (thumbnail NCC, any shift
    with enough overlap, any of the 8 transforms). With ``query_only`` only the
    first image is compared against the rest."""
    import torch

    if len(idx) < 2:
        return []
    device = _torch_device(cfg.device)
    S = cfg.thumb_size
    eng = _NCCEngine(S, cfg.min_overlap, device)
    thumbs = np.stack([texture_image(images[i], S)[0] for i in idx])
    with torch.no_grad():
        x = torch.from_numpy(thumbs).to(device)
        Fa, Fasq = eng.spectra(x)
        Sa, Saa = eng.a_terms(Fa, Fasq)
        V = torch.stack([_dihedral_t(x, k) for k in range(8)], dim=1)
        Fb, Fbsq = eng.spectra(V)
        Sb, Sbb = eng.b_terms(Fb, Fbsq)
        del Fasq, Fbsq, V
        N = len(idx)
        score = torch.full((N, N), -1.0, device=device)
        chunk = max(1, int(4.0e7 // (8 * eng.P * eng.P)))
        for i in ([0] if query_only else range(N)):
            for j0 in range(0, N, chunk):
                m = eng.ncc(Fa[i], Sa[i], Saa[i], Fb[j0:j0 + chunk], Sb[j0:j0 + chunk], Sbb[j0:j0 + chunk])
                score[i, j0:j0 + chunk] = m.flatten(-3).amax(-1)
        score.fill_diagonal_(-1.0)
        score = torch.maximum(score, score.T).cpu().numpy()
    k = min(cfg.top_k, N - 1)
    if query_only:
        top0 = np.argsort(-score[0])[:k]
        return [(idx[0], idx[int(b)]) for b in top0]
    top = np.argpartition(-score, k - 1, axis=1)[:, :k]
    cand = {(min(a, int(b)), max(a, int(b))) for a in range(N) for b in top[a]}
    return sorted((idx[a], idx[b]) for a, b in cand)


def _alignments(images, pairs: List[Tuple[int, int]], cfg: DuplicateConfig, n_best: int = 3):
    """Candidate alignments per pair at native resolution: the best shifts by
    raw NCC and by overlap-weighted NCC (so tiny overlaps cannot win alone)."""
    import torch

    out: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    if not pairs:
        return out
    shapes = {images[i].shape for p in pairs for i in p}
    first = images[pairs[0][0]].shape
    if len(shapes) != 1 or first[0] != first[1]:
        S, scale = 192, True  # mixed or non-square sizes: compare on a common grid
    else:
        S, scale = first[0], False
    device = _torch_device(cfg.device)
    eng = _NCCEngine(S, cfg.min_overlap, device)
    tex: Dict[int, np.ndarray] = {}
    for p in pairs:
        for i in p:
            if i not in tex:
                tex[i] = texture_image(images[i], S)[0]
    batch = max(1, int(3.0e7 // (8 * eng.P * eng.P)))
    with torch.no_grad():
        for s0 in range(0, len(pairs), batch):
            chunk = pairs[s0:s0 + batch]
            A = torch.from_numpy(np.stack([tex[a] for a, _ in chunk])).to(device)
            B = torch.from_numpy(np.stack([tex[b] for _, b in chunk])).to(device)
            Bv = torch.stack([_dihedral_t(B, k) for k in range(8)], dim=1)
            Fa, Fasq = eng.spectra(A)
            Sa, Saa = eng.a_terms(Fa, Fasq)
            Fb, Fbsq = eng.spectra(Bv)
            Sb, Sbb = eng.b_terms(Fb, Fbsq)
            m = eng.ncc(Fa[:, None], Sa[:, None], Saa[:, None], Fb, Sb, Sbb)
            top_raw = m.flatten(1).topk(n_best, dim=1).indices.cpu().numpy()
            top_wtd = (m * eng.weight).flatten(1).topk(n_best, dim=1).indices.cpu().numpy()
            for (a, b), r1, r2 in zip(chunk, top_raw, top_wtd):
                al = []
                for ix in list(r1) + list(r2):
                    k, rem = divmod(int(ix), eng.P * eng.P)
                    dy, dx = eng.index_to_shift(rem)
                    if scale:
                        f = images[a].shape[0] / S
                        dy, dx = round(dy * f), round(dx * f)
                    al.append((k, dy, dx))
                out[(a, b)] = list(dict.fromkeys(al))
    return out


# --------------------------------------------------------------------------
# review file
# --------------------------------------------------------------------------
def load_review(path) -> Dict[str, set]:
    """Human decisions: ``same`` and ``distinct`` lists of [name_a, name_b]
    (file names, so the file stays valid if the dataset folder moves)."""
    out = {"same": set(), "distinct": set()}
    if not path:
        return out
    path = Path(path)
    if not path.is_file():
        return out
    import yaml

    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    for key in ("same", "distinct"):
        for pair in data.get(key) or []:
            if len(pair) != 2:
                raise ValueError(f"{path}: every entry under '{key}' must be a pair")
            out[key].add(frozenset(Path(p).name for p in pair))
    clash = out["same"] & out["distinct"]
    if clash:
        raise ValueError(f"{path}: pairs listed as both same and distinct: {sorted(map(sorted, clash))}")
    return out


# --------------------------------------------------------------------------
# union-find and main entry point
# --------------------------------------------------------------------------
class _UnionFind:
    def __init__(self, items: Sequence[str]):
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self) -> List[List[str]]:
        comp: Dict[str, List[str]] = {}
        for x in self.parent:
            comp.setdefault(self.find(x), []).append(x)
        return sorted((sorted(g) for g in comp.values() if len(g) > 1), key=lambda g: g[0])


def _cache_key(paths: Sequence[str], cfg: DuplicateConfig) -> str:
    h = hashlib.sha256()
    h.update(DETECTOR_VERSION.encode())
    params = {k: v for k, v in asdict(cfg).items() if k not in ("cache_dir", "device", "review_file")}
    h.update(json.dumps(params, sort_keys=True).encode())
    for p in sorted(paths):
        st = Path(p).stat()
        h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _evaluate_candidates(images, names, pairs, cfg) -> Dict[Tuple[int, int], PairResult]:
    out = {}
    aligns = _alignments(images, pairs, cfg)
    for (a, b), al in aligns.items():
        best = None
        for k, dy, dx in al:
            ev = evaluate_alignment(images[a], images[b], k, dy, dx, cfg)
            if _better(ev, best, cfg):
                best = ev
        kind, evidence = decide(best, cfg)
        out[(a, b)] = PairResult(names[a], names[b], kind, evidence,
                                 round(best["exact_fraction"], 4),
                                 None if best["grain"] is None else round(best["grain"], 4),
                                 best["tiles"], best["overlap_px"], DIHEDRAL_NAMES[best["k"]],
                                 (best["dy"], best["dx"]))
    return out


def detect_duplicates(paths: Sequence[str], cfg: DuplicateConfig | None = None,
                      use_cache: bool = True, verbose: bool = False,
                      cache_only: bool = False) -> DuplicateReport:
    """Find groups of identical, overlapping and re-processed copies.

    With ``cache_only`` the result must come from the cache; if none matches,
    :class:`DuplicateCacheMissing` is raised instead of running the detection
    (which takes minutes). The cache key includes the path strings, so the
    caller must name the images exactly as the run that built the cache did.
    """
    cfg = cfg or DuplicateConfig()
    paths = [str(p) for p in paths]
    review = load_review(cfg.review_file)
    cache_file = None
    if use_cache and cfg.cache_dir:
        cache_file = Path(cfg.cache_dir) / f"duplicates_{_cache_key(paths, cfg)}.json"
        if cache_file.exists():
            with open(cache_file) as fh:
                raw = json.load(fh)
            pairs = [PairResult(**{**p, "shift": tuple(p["shift"]), "decided_by": "detector"})
                     for p in raw["pairs"]]
            return _finalize(paths, pairs, raw["n_candidates"], cfg, review)
    if cache_only:
        raise DuplicateCacheMissing(
            f"No cached near-duplicate result for these {len(paths)} images "
            f"(looked for {cache_file}). Run scripts/dataset_report.py first.")

    images = [_read_gray(p) for p in paths]
    ok = [i for i, im in enumerate(images) if im is not None]
    results: Dict[Tuple[int, int], PairResult] = {}

    # Identical files.
    by_hash: Dict[str, List[int]] = {}
    for i in ok:
        by_hash.setdefault(pixel_hash(images[i]), []).append(i)
    for idx in by_hash.values():
        for x in idx[1:]:
            a, b = sorted((idx[0], x))
            results[(a, b)] = PairResult(paths[a], paths[b], "same", "identical-file", 1.0, 1.0, 0,
                                         int(images[a].size), "identity", (0, 0))

    # E1: exact overlaps from pixel-run voting.
    ex = exact_overlap_candidates(images, ok, cfg)
    if verbose:
        print(f"[dups] pixel-run voting proposed {len(ex)} alignments")
    for (i, j), (k, dy, dx) in ex.items():
        a, b = sorted((i, j))
        if (a, b) in results and results[(a, b)].kind == "same":
            continue
        best = None
        for ddy in (-1, 0, 1):
            for ddx in (-1, 0, 1):
                ev = evaluate_alignment(images[i], images[j], k, dy + ddy, dx + ddx, cfg)
                if ev["exact"] and (best is None or ev["exact_fraction"] > best["exact_fraction"]):
                    best = ev
        if best is not None:
            results[(a, b)] = PairResult(paths[i], paths[j], "same", "pixel-identical",
                                         round(best["exact_fraction"], 4),
                                         None if best["grain"] is None else round(best["grain"], 4),
                                         best["tiles"], best["overlap_px"], DIHEDRAL_NAMES[best["k"]],
                                         (best["dy"], best["dx"]))

    # E2: re-processed copies among each image's most correlated partners.
    textured = [i for i in ok if not texture_image(images[i], cfg.thumb_size)[1]]
    cand = [p for p in ncc_candidates(images, textured, cfg)
            if not (p in results and results[p].kind == "same")]
    if verbose:
        print(f"[dups] verifying {len(cand)} correlation candidates at native resolution")
    results.update(_evaluate_candidates(images, paths, cand, cfg))

    pairs = list(results.values())
    n_cand = len(ex) + len(cand)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as fh:
            json.dump({"pairs": [asdict(p) for p in pairs], "n_candidates": n_cand}, fh)
    return _finalize(paths, pairs, n_cand, cfg, review)


def _finalize(paths, pairs: List[PairResult], n_candidates: int, cfg: DuplicateConfig,
              review: Dict[str, set]) -> DuplicateReport:
    uf = _UnionFind(paths)
    for p in pairs:
        key = frozenset((Path(p.a).name, Path(p.b).name))
        if p.evidence != "identical-file":
            if key in review["same"]:
                p.decided_by = "review:same"
            elif key in review["distinct"]:
                p.decided_by = "review:distinct"
        if p.evidence == "identical-file" or p.decided_by == "review:same" or (
                p.decided_by == "detector" and p.kind in ("same", "review")):
            uf.union(p.a, p.b)
    names = {Path(p).name: p for p in paths}
    seen = {frozenset((Path(p.a).name, Path(p.b).name)) for p in pairs}
    for key in review["same"] - seen:
        a, b = sorted(key)
        if a in names and b in names:
            uf.union(names[a], names[b])
            pairs.append(PairResult(names[a], names[b], "same", "review", -1.0, None, 0, 0,
                                    "n/a", (0, 0), decided_by="review:same"))
    order = {"same": 0, "review": 1, "unverified": 2, "distinct": 3}
    pairs.sort(key=lambda p: (order[p.kind], -(p.grain if p.grain is not None else -2), p.a, p.b))
    return DuplicateReport(groups=uf.groups(), pairs=pairs, n_images=len(paths),
                           n_candidates=n_candidates, config=asdict(cfg))


def _inverse_dihedral(k: int) -> int:
    """Index of the inverse transform: rotations invert to the opposite
    rotation, every flip (with or without rotation) is its own inverse."""
    return k if k >= 4 else (4 - k) % 4


def native_exact_matches(query: np.ndarray, refs: Sequence[np.ndarray], names: Sequence[str],
                         cfg: DuplicateConfig) -> Dict[str, PairResult]:
    """Pixel-identical matches of ``query`` at its own scale inside each reference.

    This is E1 for one external image: horizontal pixel runs of the query (all 8
    flips/rotations, subsampled) are hashed once, every reference is scanned at
    full density, matching runs vote for an alignment, and the best alignment
    is verified with :func:`evaluate_alignment`. It finds crops of a dataset
    image, which resizing the query to the reference size would destroy.
    Results are keyed by reference name; ``transform`` names how the reference
    is transformed to match the query.
    """
    L, s = cfg.segment_len, cfg.segment_stride
    if query.shape[0] < L or query.shape[1] < L:
        return {}
    R = np.random.default_rng(12345).integers(1, 2**63 - 1, size=L, dtype=np.uint64) | np.uint64(1)
    keys, meta = [], []
    for k in range(8):
        h, r, c = _run_hashes(dihedral(query, k), L, s, s, cfg.min_segment_range, R)
        keys.append(h)
        meta.append(np.stack([np.full_like(r, k), r, c], 1).astype(np.int32))
    keys = np.concatenate(keys)
    if len(keys) == 0:
        return {}
    meta = np.concatenate(meta)
    order = np.argsort(keys, kind="stable")
    keys, meta = keys[order], meta[order]
    uniq, start, counts = np.unique(keys, return_index=True, return_counts=True)
    ok = counts <= cfg.max_key_occurrences
    uniq, start, counts = uniq[ok], start[ok], counts[ok]
    if len(uniq) == 0:
        return {}

    out: Dict[str, PairResult] = {}
    for ref, name in zip(refs, names):
        if ref is None or ref.shape[1] < L:
            continue
        h, r, c = _run_hashes(ref, L, 1, 1, cfg.min_segment_range, R)
        if len(h) == 0:
            continue
        pos = np.clip(np.searchsorted(uniq, h), 0, len(uniq) - 1)
        hits = np.nonzero(uniq[pos] == h)[0]
        if len(hits) < cfg.min_votes:
            continue
        votes: Dict[Tuple[int, int, int], int] = {}
        for t in hits:
            p = pos[t]
            for e in range(start[p], start[p] + counts[p]):
                k, rq, cq = meta[e]
                key = (int(k), int(rq - r[t]), int(cq - c[t]))
                votes[key] = votes.get(key, 0) + 1
        (k, dy, dx), v = max(votes.items(), key=lambda kv: kv[1])
        if v < cfg.min_votes:
            continue
        best = None
        for ddy in (-1, 0, 1):
            for ddx in (-1, 0, 1):
                # reference = a, query transformed by k = b (the query side was hashed)
                ev = evaluate_alignment(ref, query, k, dy + ddy, dx + ddx, cfg)
                if ev["exact"] and (best is None or ev["exact_fraction"] > best["exact_fraction"]):
                    best = ev
        if best is not None:
            out[name] = PairResult("<query>", name, "same", "pixel-identical",
                                   round(best["exact_fraction"], 4),
                                   None if best["grain"] is None else round(best["grain"], 4),
                                   best["tiles"], best["overlap_px"],
                                   DIHEDRAL_NAMES[_inverse_dihedral(best["k"])],
                                   (best["dy"], best["dx"]))
    return out


def _match_order(p: PairResult):
    # Strongest evidence first; among pixel-identical matches the largest
    # overlap (the source image itself rather than a neighbouring crop); among
    # the rest the highest grain score.
    return (-_RANK[p.kind], -p.overlap_px if p.evidence == "pixel-identical" else 0,
            -(p.grain if p.grain is not None else -2))


def match_against(query: np.ndarray, reference_paths: Sequence[str],
                  cfg: DuplicateConfig | None = None, top_n: int = 3) -> List[PairResult]:
    """Compare one external image (for example an upload) with a reference set.

    Two searches are combined, both under all 8 flips/rotations and any shift:

    * the query resized to the reference size, through the correlation search
      and E1/E2 verification, so a whole-image copy at another resolution or
      with changed brightness can still be recognized;
    * if the query is not larger than the references, pixel identity at the
      query's own scale (:func:`native_exact_matches`), so a crop of a dataset
      image is recognized too.

    Returns the ``top_n`` partners ordered by evidence (same, review,
    unverified, distinct), strongest first.
    """
    cfg = cfg or DuplicateConfig()
    q0 = query if query.ndim == 2 else cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    refs = [_read_gray(p) for p in reference_paths]
    keep = [i for i, r in enumerate(refs) if r is not None]
    if not keep:
        return []
    ref_imgs = [refs[i] for i in keep]
    ref_names = [str(reference_paths[i]) for i in keep]
    shape = ref_imgs[0].shape
    q = q0 if q0.shape == shape else cv2.resize(q0, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    if texture_image(q, cfg.thumb_size)[1]:
        return []
    images = [q] + ref_imgs
    names = ["<query>"] + ref_names
    cand = ncc_candidates(images, list(range(len(images))), cfg, query_only=True)
    found = {p.b: p for p in _evaluate_candidates(images, names, cand, cfg).values()}
    if q0.shape[0] <= shape[0] and q0.shape[1] <= shape[1]:
        for name, p in native_exact_matches(q0, ref_imgs, ref_names, cfg).items():
            old = found.get(name)
            if old is None or _match_order(p) < _match_order(old):
                found[name] = p
    out = sorted(found.values(), key=_match_order)
    return out[:top_n]
