"""Shared-content checks: vectorized voting, E1 texture guard, masked NCC screen."""

from typing import Dict, Tuple

import cv2
import numpy as np
import pytest

from idefect.config import DuplicateConfig
from idefect.data import duplicates as dup
from idefect.data.similarity_screen import (_dihedral, band_thumb, copy_variants, pair_score,
                                            similarity_matrix)


def _tex(seed, shape=(120, 120), sigma=1.0):
    rng = np.random.default_rng(seed)
    x = cv2.GaussianBlur((rng.random(shape) * 255).astype(np.uint8), (0, 0), sigma)
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)


def _loop_votes(images, idx, cfg) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
    """The original Python-loop implementation, kept as the reference."""
    L, s = cfg.segment_len, cfg.segment_stride
    R = np.random.default_rng(12345).integers(1, 2**63 - 1, size=L, dtype=np.uint64) | np.uint64(1)
    keys, meta = [], []
    for j in idx:
        for k in range(8):
            h, r, c = dup._run_hashes(dup.dihedral(images[j], k), L, s, s, cfg.min_segment_range, R)
            keys.append(h)
            meta.append(np.stack([np.full_like(r, j), np.full_like(r, k), r, c], 1).astype(np.int32))
    keys, meta = np.concatenate(keys), np.concatenate(meta)
    order = np.argsort(keys, kind="stable")
    keys, meta = keys[order], meta[order]
    uniq, start, counts = np.unique(keys, return_index=True, return_counts=True)
    ok = counts <= cfg.max_key_occurrences
    uniq, start, counts = uniq[ok], start[ok], counts[ok]
    votes = {}
    for i in idx:
        h, r, c = dup._run_hashes(images[i], L, 1, 1, cfg.min_segment_range, R)
        pos = np.clip(np.searchsorted(uniq, h), 0, len(uniq) - 1)
        for t in np.nonzero(uniq[pos] == h)[0]:
            p = pos[t]
            for e in range(start[p], start[p] + counts[p]):
                j, k, rb, cb = meta[e]
                if j == i:
                    continue
                key = (i, int(j), int(k), int(rb - r[t]), int(cb - c[t]))
                votes[key] = votes.get(key, 0) + 1
    best = {}
    for (i, j, k, dy, dx), v in votes.items():
        if v >= cfg.min_votes and ((i, j) not in best or v > best[(i, j)][0]):
            best[(i, j)] = (v, k, dy, dx)
    return {pair: (k, dy, dx) for pair, (v, k, dy, dx) in best.items()}


def _overlap_set():
    big = _tex(1, (200, 200))
    imgs = [big[:120, :120], np.ascontiguousarray(np.rot90(big[50:170, 40:160])),
            np.ascontiguousarray(np.fliplr(big[80:200, 70:190])), _tex(2), _tex(3, (90, 140))]
    return [np.ascontiguousarray(x) for x in imgs]


def test_vectorized_votes_equal_loop():
    cfg = DuplicateConfig()
    imgs = _overlap_set()
    idx = list(range(len(imgs)))
    ref = _loop_votes(imgs, idx, cfg)
    got = dup.exact_overlap_candidates(imgs, idx, cfg)
    assert got == ref and list(got) == list(ref)
    assert (0, 1) in got and (0, 2) in got


def test_exact_pairs_find_crops_under_transforms():
    pairs = dup.exact_overlap_pairs(_overlap_set(), DuplicateConfig())
    found = {(p["a"], p["b"]) for p in pairs}
    assert {(0, 1), (0, 2), (1, 2)} <= found
    assert not any(3 in pr or 4 in pr for pr in found)


def test_texture_guard_rejects_flat_overlap():
    # two different images sharing only a large black area and one small textured spot
    a = np.zeros((150, 150), np.uint8)
    b = np.zeros((150, 150), np.uint8)
    spot = _tex(5, (14, 14))
    a[10:24, 10:24] = spot
    b[10:24, 10:24] = spot
    a[100:140, 100:140] = _tex(6, (40, 40))
    b[100:140, 100:140] = _tex(7, (40, 40))
    cfg = DuplicateConfig(min_votes=1)
    assert dup.exact_overlap_pairs([a, b], cfg) == []


def _brute(a, b, mo):
    best = -9.0
    ha, wa = a.shape
    hb, wb = b.shape
    for dy in range(-hb + 1, ha):
        for dx in range(-wb + 1, wa):
            y0, y1, x0, x1 = max(0, -dy), min(hb, ha - dy), max(0, -dx), min(wb, wa - dx)
            if y1 <= y0 or x1 <= x0:
                continue
            pb, pa = b[y0:y1, x0:x1], a[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            if pa.size < mo * min(a.size, b.size) - 0.5:
                continue
            p, q = pa - pa.mean(), pb - pb.mean()
            den = np.sqrt((p * p).sum() * (q * q).sum())
            if den > 1e-6:
                best = max(best, float((p * q).sum() / den))
    return best


@pytest.mark.parametrize("device", ["cpu"])
def test_masked_ncc_equals_brute_force(device):
    rng = np.random.default_rng(0)
    ts = [rng.normal(size=(int(rng.integers(8, 12)), int(rng.integers(8, 12)))).astype(np.float32)
          for _ in range(4)]
    r = similarity_matrix(ts, (0, 2, 4, 6), 0.5, device)
    for i in range(4):
        for j in range(i + 1, 4):
            ref = max(_brute(ts[i], _dihedral(ts[j], k), 0.5) for k in (0, 2, 4, 6))
            assert abs(ref - r["score"][i, j]) < 1e-4
            assert r["score"][i, j] == r["score"][j, i]
    assert (np.diag(r["score"]) == -1).all()


def test_planted_copies_score_high_and_distinct_low():
    img = _tex(11, (256, 256), sigma=1.5)
    other = _tex(12, (256, 256), sigma=1.5)
    rng = np.random.default_rng(0)
    for name, cp in copy_variants(img, rng, square=True).items():
        assert pair_score(img, cp, 2.0, tuple(range(8)), device="cpu")[0] > 0.7, name
    assert pair_score(img, other, 2.0, tuple(range(8)), device="cpu")[0] < 0.3
    assert band_thumb(np.full((64, 64), 128, np.uint8), 2.0).std() == 0
