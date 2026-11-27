"""Phase 4 dataset report: integrity, shared content, leakage-safe splits.

``build_phase4_report`` (``scripts/phase4_dataset_report.py``) runs, in order,
and stops at the first failure:

1. both manifests against the files on disk (missing, modified, unexpected);
2. the loaders' strict layout checks and the documented counts;
3. the three-layer shared-content check per dataset (per DAGM class), with the
   planted-copy calibration;
4. KolektorSDD: part relations, official versus corrected folds, validation
   parts; DAGM: groups, the official Train/Test boundary, validation images;
5. the split files in ``data/splits``: written once, afterwards recomputed
   and compared, so a split can never change silently.

Outputs: ``experiments/results/phase4_datasets.json`` and the split files.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from idefect.config import Config
from idefect.data.archives import read_sha256_list, verify_tree
from idefect.data.dagm import load_dagm, validation_split
from idefect.data.kolektor import (DatasetError, corrected_folds, load_kolektorsdd,
                                   load_official_folds, mask_iou, part_relations, read_image,
                                   read_mask, split_report)
from idefect.data.phase4_leakage import (content_check, groups_from_pairs, load_review_decisions,
                                         review_decided, review_needed, same_pairs)
from idefect.data.similarity_screen import SHAPE_PRESERVING

KOLEKTOR_COUNTS = {"parts": 50, "images": 399, "defective": 52}
DAGM_COUNTS = {"images": 16100, "defective": 2100}
SPLIT_DIR = Path("data/splits")


def _check_tree(root: str, manifest: str, suffixes, ignore=()) -> Dict[str, str]:
    expected = read_sha256_list(manifest)
    rep = verify_tree(root, expected, suffixes, ignore)
    bad = {k: rep[k][:5] for k in ("missing", "modified", "unexpected") if rep[k]}
    if bad:
        raise DatasetError(f"{root} does not match {manifest}: {bad}. Run scripts/prepare_phase4_data.py.")
    return expected


def _write_or_compare(path: Path, obj: Dict) -> str:
    text = json.dumps(obj, indent=1, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text() != text:
            raise DatasetError(f"{path} differs from the split computed now. A split must never change "
                               "silently: find out why (data, settings or code changed), then delete the "
                               "file to write the new split deliberately.")
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return "written"


def kolektorsdd_report(cfg: Config, verbose: bool = True) -> Dict:
    kc = cfg.kolektorsdd
    digests = _check_tree(kc.root, kc.manifest, (".jpg", ".bmp", ".pyb", ".png"))
    samples = load_kolektorsdd(kc.root)
    counts = {"parts": len({s.part for s in samples}), "images": len(samples),
              "defective": sum(s.defective for s in samples)}
    if counts != KOLEKTOR_COUNTS:
        raise DatasetError(f"KolektorSDD counts {counts} differ from {KOLEKTOR_COUNTS}")
    root = Path(kc.root)
    ids = [s.id for s in samples]
    rel_paths = [Path(s.image).relative_to(root).as_posix() for s in samples]
    images = [read_image(s.image) for s in samples]
    content = content_check(images, ids, [digests[p] for p in rel_paths], kc.screen, cfg.data.duplicates,
                            SHAPE_PRESERVING, "kolektorsdd", cfg.data.duplicates.cache_dir, verbose=verbose)
    if not content["calibration_ok"]:
        raise DatasetError(f"KolektorSDD: planted copies would be missed: {content['planted_missed'][:5]}")
    pairs = same_pairs(content)
    relations = part_relations(samples, pairs)
    official = load_official_folds(kc.splits_file)
    folds = corrected_folds(official, relations, kc.duplicate_policy, kc.val_parts_frac, kc.split_seed)
    rep = split_report(samples, official, relations, folds)

    by_id = {s.id: s for s in samples}
    copies = []
    for r in relations:
        if r.kind != "copy":
            continue
        a, b = r.parts[0], r.parts[1]
        for s in (x for x in samples if x.part == a):
            other = by_id.get(f"{b}/Part{s.index}")
            if other is None or not (s.defective or other.defective):
                continue
            iou = mask_iou(read_mask(s.mask), read_mask(other.mask))
            copies.append({"a": s.id, "b": other.id, "defect_px": [s.defect_px, other.defect_px],
                           "mask_iou": None if iou is None else round(iou, 4)})
    cross_part = [p for p in content["pairs_same_or_review"]
                  if p["a"].split("/")[0] != p["b"].split("/")[0]]
    split_file = {"policy": folds.policy, "dropped_parts": folds.dropped,
                  "relations": [{"parts": r.parts, "kind": r.kind, "evidence": r.evidence} for r in relations],
                  "folds": [{"train": tr, "val": va, "test": te}
                            for tr, va, te in zip(folds.train, folds.val, folds.test)],
                  "val_parts_frac": kc.val_parts_frac, "split_seed": kc.split_seed}
    status = _write_or_compare(SPLIT_DIR / "kolektorsdd_folds.json", split_file)
    sizes = Counter((s.height, s.width) for s in samples)
    return {"counts": counts, "image_heights": [min(h for h, _ in sizes), max(h for h, _ in sizes)],
            "image_widths": sorted({w for _, w in sizes}), "n_sizes": len(sizes),
            "content": content, "cross_part_pairs": cross_part, "copy_mask_agreement": copies,
            "splits": rep, "split_file": status}


def dagm_report(cfg: Config, verbose: bool = True) -> Dict:
    dc = cfg.dagm
    digests = _check_tree(dc.root, dc.manifest, (".png", ".txt", ".bmp", ".jpg"), ("zips",))
    samples = load_dagm(dc.root, dc.classes)
    if dc.classes == list(range(1, 11)):
        counts = {"images": len(samples), "defective": sum(s.defective for s in samples)}
        if counts != DAGM_COUNTS:
            raise DatasetError(f"DAGM counts {counts} differ from {DAGM_COUNTS}")
    root = Path(dc.root)
    decisions = load_review_decisions(dc.review_file, "dagm")
    known = {s.id for s in samples}
    in_scope = {fs for key in ("same", "distinct") for fs in decisions[key]
                if all(int(x.split("/")[0][5:]) in dc.classes for x in fs if x.startswith("Class"))}
    unknown = sorted(x for fs in in_scope for x in fs if x not in known)
    if unknown:
        raise DatasetError(f"{dc.review_file}: unknown DAGM image ids {unknown[:5]}")
    flagged = set()
    per_class, all_groups, crossing = {}, [], []
    for c in dc.classes:
        sub = [s for s in samples if s.cls == c]
        ids = [s.id for s in sub]
        dig = [digests[Path(s.image).relative_to(root).as_posix()] for s in sub]
        images = [read_image(s.image) for s in sub]
        content = content_check(images, ids, dig, dc.screen, cfg.data.duplicates, tuple(range(8)),
                                f"dagm_class{c}", cfg.data.duplicates.cache_dir, seed=c, verbose=verbose)
        del images
        if not content["calibration_ok"]:
            raise DatasetError(f"DAGM Class{c}: planted copies would be missed: {content['planted_missed'][:5]}")
        flagged |= {frozenset((p["a"], p["b"])) for p in content["pairs_same_or_review"] if p["kind"] == "review"}
        groups = groups_from_pairs(ids, same_pairs(content, decisions, ids))
        split_of = {s.id: s.split for s in sub}
        cross = [g for g in groups if len({split_of[i] for i in g}) > 1]
        crossing += cross
        all_groups += [g for g in groups if g not in cross]
        cnt = Counter((s.split, s.defective) for s in sub)
        per_class[c] = {"train": [cnt[("Train", False)], cnt[("Train", True)]],
                        "test": [cnt[("Test", False)], cnt[("Test", True)]],
                        "groups": groups, "groups_crossing_train_test": cross,
                        "review": review_needed(content, decisions),
                        "review_decided": review_decided(content, decisions), "content": content}
        if verbose:
            print(f"[DAGM] Class{c}: {len(content['exact'])} exact pairs, "
                  f"{len(content['pairs_same_or_review'])} same or review"
                  + (f" ({len(per_class[c]['review_decided'])} decided by a person)"
                     if per_class[c]["review_decided"] else "")
                  + f", max fine distinct {content['max_fine_distinct']}")
    # a "distinct" decision for a pair the check no longer flags is stale: data, settings or code changed
    stale = sorted(sorted(fs) for fs in decisions["distinct"] & in_scope if fs not in flagged)
    if crossing:
        # The official boundary itself leaks: a decision for a person, not for code.
        return {"per_class": per_class, "groups_crossing_train_test": crossing, "stale_decisions": stale,
                "split_file": "not written"}
    assign = validation_split(samples, dc.val_frac, dc.split_seed, all_groups)
    val = {f"Class{c}": sorted(s.name for s in samples if s.cls == c and assign[s.id] == "val")
           for c in dc.classes}
    decided = sorted(sorted([p["a"], p["b"]]) + [p["decision"]]
                     for c in dc.classes for p in per_class[c]["review_decided"])
    split_file = {"val_frac": dc.val_frac, "split_seed": dc.split_seed, "classes": dc.classes,
                  "groups": all_groups, "val": val, "review_decisions": decided,
                  "note": "official Test is the test set; Train minus val is the training set"}
    status = _write_or_compare(SPLIT_DIR / "dagm_split.json", split_file)
    sides = Counter((s.cls, assign[s.id], s.defective) for s in samples)
    for c in dc.classes:
        per_class[c]["split"] = {side: [sides[(c, side, False)], sides[(c, side, True)]]
                                 for side in ("train", "val", "test")}
    return {"per_class": per_class, "groups_crossing_train_test": [], "stale_decisions": stale,
            "split_file": status}


def build_phase4_report(cfg: Config, out_dir: str | None = None, only: str | None = None,
                        verbose: bool = True) -> Dict:
    out: Dict = {}
    if only in (None, "kolektorsdd"):
        out["kolektorsdd"] = kolektorsdd_report(cfg, verbose)
    if only in (None, "dagm"):
        out["dagm"] = dagm_report(cfg, verbose)
    out_dir = Path(out_dir or cfg.output.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"phase4_datasets{'' if only is None else '_' + only}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    out["_path"] = str(path)
    return out


def problems(report: Dict) -> List[str]:
    """Issues needing a person (empty list: nothing open)."""
    out = []
    k = report.get("kolektorsdd")
    if k:
        for f in k["splits"]["corrected"]:
            if f["crossing"]:
                out.append(f"KolektorSDD fold {f['fold']}: relations still cross {f['crossing']}")
        for p in k["content"]["pairs_same_or_review"]:
            if p["kind"] == "review":
                out.append(f"KolektorSDD review pair {p['a']} ~ {p['b']} (fine {p['fine']})")
    d = report.get("dagm")
    if d:
        for g in d["groups_crossing_train_test"]:
            out.append(f"DAGM group crosses the official Train/Test boundary: {g}")
        for pr in d.get("stale_decisions", []):
            out.append(f"DAGM review decision no longer matches a flagged pair: {pr}")
        for c, pc in d["per_class"].items():
            for p in pc["review"]:
                out.append(f"DAGM Class{c} review pair {p['a']} ~ {p['b']} (fine {p['fine']})")
    return out
