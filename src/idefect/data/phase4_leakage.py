"""Shared-content checks for KolektorSDD and DAGM: what may never cross a split.

Three layers, each with its own evidence:

1. Identical files and identical decoded pixels (hashes).
2. E1, pixel identity under any flip, rotation or shift, for any image shape
   (``duplicates.exact_overlap_pairs``), with a texture guard so flat areas
   cannot count as shared content.
3. Re-processed or re-captured copies: a coarse masked-NCC screen over all
   pairs (``similarity_screen``), then a fine-resolution score for every
   candidate (each image's ``top_k`` coarse partners plus every pair at or
   above ``candidate_score``). The fine score decides: ``same`` and above is
   the same content, ``review`` to ``same`` is grouped too and shown to a
   person, below is distinct.

Layer 3 is calibrated on the data itself. For ``planted_sources`` images,
known re-processed copies are made (``similarity_screen.copy_variants``) and
pushed through the same pipeline: each copy must become a candidate (coarse
score at or above ``candidate_score``, or above the source's ``top_k``-th best
real partner) and must reach the ``same`` threshold at fine scale. If any
planted copy would be missed, ``calibration_ok`` is False and the caller stops.

Results are cached under ``experiments/cache``, keyed by the SHA-256 of every
image involved, the settings and ``CHECK_VERSION``. The cache file is written
in one step (temporary file, then rename), so a stopped run leaves either the
complete result or nothing.

``progress``, ``gate`` and ``e1_workers`` only change how the work is run
(reporting, resource reservations, parallel E1), never the result, so they are
not part of the cache key.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from idefect.config import ContentScreenConfig, DuplicateConfig
from idefect.data.duplicates import exact_overlap_pairs
from idefect.data.similarity_screen import (DIHEDRAL, band_thumb, copy_variants, pair_score,
                                            similarity_matrix)

CHECK_VERSION = "1"


def _cache_file(cache_dir: str, tag: str, digests: Sequence[str], sc: ContentScreenConfig,
                dup: DuplicateConfig, transforms: Sequence[int]) -> Path:
    h = hashlib.sha256()
    h.update(CHECK_VERSION.encode())
    h.update(json.dumps({"screen": asdict(sc), "dup": {k: v for k, v in asdict(dup).items()
                                                        if k not in ("cache_dir", "device", "review_file")},
                         "transforms": list(transforms)}, sort_keys=True).encode())
    for d in digests:
        h.update(d.encode())
    return Path(cache_dir) / f"phase4_content_{tag}_{h.hexdigest()[:16]}.json"


def content_check(images: Sequence[np.ndarray], ids: Sequence[str], digests: Sequence[str],
                  sc: ContentScreenConfig, dup: DuplicateConfig, transforms: Sequence[int],
                  tag: str, cache_dir: str = "experiments/cache", seed: int = 0,
                  verbose: bool = False, progress=None, gate=None, e1_workers: int = 1) -> Dict:
    """Run all three layers on one image set; returns a JSON-ready dict.

    ``progress(stage, done, total)`` reports the stages e1, coarse (in pairs),
    fine and planted. ``gate(stage, **info)`` returns a context manager
    entered around the stages e1 and coarse (``info`` has ``thumbs`` and
    ``transforms`` for coarse), so a caller can reserve memory first.
    """
    cache = _cache_file(cache_dir, tag, digests, sc, dup, transforms) if cache_dir else None
    if cache is not None and cache.is_file():
        with open(cache) as fh:
            return json.load(fh)
    report = progress or (lambda stage, done, total: None)
    gate = gate or (lambda stage, **info: nullcontext())
    N = len(images)
    with gate("e1"):
        exact = exact_overlap_pairs(images, replace(dup, segment_stride=sc.exact_stride), verbose=verbose,
                                    workers=e1_workers, progress=lambda d, t: report("e1", d, t))
    exact_out = [{**e, "a": ids[e["a"]], "b": ids[e["b"]], "shift": list(e["shift"])} for e in exact]

    thumbs = [band_thumb(im, sc.coarse_factor) for im in images]
    n_pairs = N * (N - 1) // 2
    with gate("coarse", thumbs=thumbs, transforms=transforms):
        coarse = similarity_matrix(
            thumbs, transforms, sc.min_overlap,
            progress=lambda q, t: report("coarse", q * (2 * N - q - 1) // 2, n_pairs))
    S = coarse["score"]
    k = min(sc.top_k, N - 1)
    top = np.argpartition(-S, k - 1, axis=1)[:, :k]
    cand = {(min(i, int(j)), max(i, int(j))) for i in range(N) for j in top[i]}
    hi = np.argwhere(np.triu(S >= sc.candidate_score, 1))
    cand |= {(int(a), int(b)) for a, b in hi}
    if verbose:
        print(f"[screen] {tag}: {len(cand)} candidates for fine verification")
    pairs = []
    for n_done, (a, b) in enumerate(sorted(cand), 1):
        f, t, dy, dx = pair_score(images[a], images[b], sc.fine_factor, transforms, sc.min_overlap)
        kind = "same" if f >= sc.same else "review" if f >= sc.review else "distinct"
        pairs.append({"a": ids[a], "b": ids[b], "coarse": round(float(S[a, b]), 4), "fine": round(f, 4),
                      "kind": kind, "transform": DIHEDRAL[t], "shift": [dy, dx]})
        report("fine", n_done, len(cand))
    iu = np.triu_indices(N, 1)
    allc = S[iu]
    kth = np.sort(S, axis=1)[:, -k]          # each image's k-th best real partner

    # Planted copies through the same pipeline.
    rng = np.random.default_rng(seed)
    src = sorted(rng.choice(N, size=min(sc.planted_sources, N), replace=False).tolist())
    square = all(im.shape[0] == im.shape[1] for im in images)
    planted: Dict[str, Dict[str, List[float]]] = {}
    missed = []
    for n_done, s in enumerate(src, 1):
        for name, cp in copy_variants(images[s], rng, square).items():
            c = pair_score(images[s], cp, sc.coarse_factor, transforms, sc.min_overlap)[0]
            f = pair_score(images[s], cp, sc.fine_factor, transforms, sc.min_overlap)[0]
            d = planted.setdefault(name, {"coarse": [], "fine": []})
            d["coarse"].append(round(c, 4))
            d["fine"].append(round(f, 4))
            if not ((c >= sc.candidate_score or c > kth[s]) and f >= sc.same):
                missed.append({"source": ids[s], "variant": name, "coarse": round(c, 4),
                               "kth_partner": round(float(kth[s]), 4), "fine": round(f, 4)})
        report("planted", n_done, len(src))
    distinct = [p["fine"] for p in pairs if p["kind"] == "distinct"]
    out = {
        "tag": tag, "n_images": N, "transforms": [DIHEDRAL[t] for t in transforms],
        "exact": exact_out,
        "n_candidates": len(pairs),
        "pairs_same_or_review": [p for p in pairs if p["kind"] != "distinct"],
        "top_distinct": sorted((p for p in pairs if p["kind"] == "distinct"),
                               key=lambda p: -p["fine"])[:10],
        "coarse_quantiles": {q: round(float(np.quantile(allc, float(q))), 4)
                             for q in ("0.5", "0.99", "0.999", "1.0")} if allc.size else {},
        "max_fine_distinct": max(distinct) if distinct else None,
        "planted": {n: {"coarse_min": min(v["coarse"]), "fine_min": min(v["fine"]),
                        "fine_median": float(np.median(v["fine"]))} for n, v in planted.items()},
        "planted_sources": [ids[s] for s in src],
        "planted_missed": missed,
        "calibration_ok": not missed,
        "settings": asdict(sc),
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.name + f".partial{os.getpid()}")
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(tmp, cache)
    return out


def load_review_decisions(path: str | Path | None, dataset: str) -> Dict[str, set]:
    """Human decisions for one dataset from the review file: {"same": {...}, "distinct": {...}}.

    The file maps a dataset name to ``same`` and ``distinct`` lists of
    [image id, image id] pairs. Anything malformed raises: unknown keys, an
    entry that is not two different ids, a pair listed twice or under both.
    """
    out: Dict[str, set] = {"same": set(), "distinct": set()}
    if not path or not Path(path).is_file():
        return out
    import yaml

    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    unknown = set(data) - {"kolektorsdd", "dagm"}
    if unknown:
        raise ValueError(f"{path}: unknown datasets {sorted(unknown)}")
    part = data.get(dataset) or {}
    unknown = set(part) - {"same", "distinct"}
    if unknown:
        raise ValueError(f"{path}: unknown keys under '{dataset}': {sorted(unknown)}")
    for key in ("same", "distinct"):
        for pair in part.get(key) or []:
            if (not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(x, str) for x in pair)
                    or pair[0] == pair[1]):
                raise ValueError(f"{path}: '{dataset}.{key}' entries must be two different image ids: {pair}")
            fs = frozenset(pair)
            if fs in out[key]:
                raise ValueError(f"{path}: pair listed twice under '{dataset}.{key}': {pair}")
            out[key].add(fs)
    clash = out["same"] & out["distinct"]
    if clash:
        raise ValueError(f"{path}: pairs listed as both same and distinct: {sorted(map(sorted, clash))}")
    return out


def same_pairs(result: Dict, decisions: Dict[str, set] | None = None,
               ids: Sequence[str] | None = None) -> List[tuple]:
    """Every pair that must stay on one side: exact, same and review, after human decisions.

    A ``distinct`` decision removes a review-band pair; it may not overrule an
    exact pair or a pair scoring ``same`` or above (that raises). A ``same``
    decision adds the pair if both images are in ``ids``.
    """
    dec = decisions or {"same": set(), "distinct": set()}
    out = []
    for e in result["exact"]:
        if frozenset((e["a"], e["b"])) in dec["distinct"]:
            raise ValueError(f"{e['a']} ~ {e['b']} share identical pixels and cannot be decided distinct")
        out.append((e["a"], e["b"]))
    for p in result["pairs_same_or_review"]:
        if frozenset((p["a"], p["b"])) in dec["distinct"]:
            if p["kind"] != "review":
                raise ValueError(f"{p['a']} ~ {p['b']} scores {p['fine']} ({p['kind']}): only pairs in the "
                                 "review band can be decided distinct")
            continue
        out.append((p["a"], p["b"]))
    if ids is not None:
        have = set(ids)
        out += [tuple(sorted(fs)) for fs in sorted(dec["same"], key=sorted) if fs <= have]
    return out


def review_decided(result: Dict, decisions: Dict[str, set] | None) -> List[Dict]:
    """Review-band pairs a person has decided, with the decision."""
    dec = decisions or {"same": set(), "distinct": set()}
    out = []
    for p in result["pairs_same_or_review"]:
        key = frozenset((p["a"], p["b"]))
        if p["kind"] == "review" and (key in dec["distinct"] or key in dec["same"]):
            out.append({**p, "decision": "distinct" if key in dec["distinct"] else "same"})
    return out


def groups_from_pairs(ids: Sequence[str], pairs: Sequence[tuple]) -> List[List[str]]:
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    comp: Dict[str, List[str]] = {}
    for x in ids:
        comp.setdefault(find(x), []).append(x)
    return sorted((sorted(g) for g in comp.values() if len(g) > 1), key=lambda g: g[0])


def review_needed(result: Dict, decisions: Dict[str, set] | None = None) -> List[Dict]:
    """Review-band pairs nobody has decided yet."""
    dec = decisions or {"same": set(), "distinct": set()}
    decided = dec["same"] | dec["distinct"]
    return [p for p in result["pairs_same_or_review"]
            if p["kind"] == "review" and frozenset((p["a"], p["b"])) not in decided]
