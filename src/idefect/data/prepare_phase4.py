"""Verify, extract and fingerprint the KolektorSDD and DAGM 2007 downloads.

Driven by ``scripts/prepare_phase4_data.py``. The run is idempotent: a second
run re-verifies every archive and every extracted file and writes nothing.

Layout produced (all under ``data/raw``, which git ignores):

    KolektorSDD/kos01 ... kos50/PartN.jpg, PartN_label.bmp
    KolektorSDD/splits/split.pyb, split_small_*.pyb, README
    DAGM/Class1 ... Class10/{Train,Test}/NNNN.PNG, Label/Labels.txt, Label/NNNN_label.PNG

Integrity manifests (committed): ``data/kolektorsdd_manifest.sha256`` and
``data/dagm_manifest.sha256``, paths relative to the dataset root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from idefect.config import Config
from idefect.data.archives import (DAGM_PUBLISHED_MD5, ArchiveError, read_sha256_list,
                                   safe_extract, verify_archive, verify_tree, write_sha256_list)

ARCHIVE_LIST = "data/phase4_archives.sha256"   # SHA-256 of the 12 downloads, relative to data/raw

# Counts measured when the downloads were first inspected; a different count
# means a different or damaged release and stops the run.
KOLEKTOR_EXPECTED_FILES = {".jpg": 399, ".bmp": 399, ".pyb": 6, "": 1}   # "" = splits/README
DAGM_EXPECTED_FILES = {c: (2 * (575 if c <= 6 else 1150)) for c in range(1, 11)}  # images per class


def _count_by_suffix(paths) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in paths:
        s = Path(p).suffix.lower()
        out[s] = out.get(s, 0) + 1
    return out


def _check_manifest(manifest: Path, members: Dict[str, str], write: bool, log) -> str:
    """The manifest must equal the archive-derived map exactly; write it once."""
    if manifest.is_file():
        recorded = read_sha256_list(manifest)
        if recorded != members:
            only_m = sorted(set(recorded) - set(members))[:5]
            only_a = sorted(set(members) - set(recorded))[:5]
            diff = sorted(k for k in set(recorded) & set(members) if recorded[k] != members[k])[:5]
            raise ArchiveError(
                f"{manifest} does not match the archives: only in manifest {only_m}, "
                f"only in archives {only_a}, different digest {diff}")
        return "matches the archives"
    if not write:
        raise ArchiveError(f"{manifest} is missing; rerun with --write-manifests to create it")
    write_sha256_list(members, manifest)
    log(f"  wrote {manifest} ({len(members)} files)")
    return "written"


def prepare_kolektorsdd(cfg: Config, raw: Path, write_manifests: bool, log=print) -> Dict:
    known = read_sha256_list(ARCHIVE_LIST)
    root = Path(cfg.kolektorsdd.root)
    archives = [("KolektorSDD/KolektorSDD.zip", r"kos\d\d", ""),
                ("KolektorSDD/KolektorSDD-training-splits.zip", r"[^/]+", "splits")]
    members: Dict[str, str] = {}
    steps: List[Dict] = []
    for rel, top, prefix in archives:
        info = verify_archive(raw / rel, known[rel])
        log(f"  {rel}: SHA-256 verified")
        res = safe_extract(raw / rel, root, top=top, prefix=prefix)
        log(f"  {rel}: {res['written']} written, {res['unchanged']} already present and identical, "
            f"{res['skipped']} skipped")
        overlap = set(members) & set(res["members"])
        if overlap:
            raise ArchiveError(f"archives overlap on {sorted(overlap)[:5]}")
        members.update(res.pop("members"))
        steps.append({**info, **res})
    counts = _count_by_suffix(members)
    if counts != KOLEKTOR_EXPECTED_FILES:
        raise ArchiveError(f"KolektorSDD file counts {counts} differ from {KOLEKTOR_EXPECTED_FILES}")
    status = _check_manifest(Path(cfg.kolektorsdd.manifest), members, write_manifests, log)
    disk = verify_tree(root, read_sha256_list(cfg.kolektorsdd.manifest),
                       scan_suffixes=(".jpg", ".bmp", ".pyb", ".png"))
    _raise_on_tree(disk, root)
    log(f"  {root}: all {disk['n_expected']} files match the manifest, no unexpected files")
    return {"archives": steps, "manifest": status, "files": len(members), "counts": counts}


def prepare_dagm(cfg: Config, raw: Path, write_manifests: bool, log=print) -> Dict:
    known = read_sha256_list(ARCHIVE_LIST)
    root = Path(cfg.dagm.root)
    members: Dict[str, str] = {}
    steps: List[Dict] = []
    for c in range(1, 11):   # always all ten: the manifest covers the whole release
        rel = f"DAGM/zips/Class{c}.zip"
        info = verify_archive(raw / rel, known[rel], md5=DAGM_PUBLISHED_MD5[f"Class{c}.zip"])
        res = safe_extract(raw / rel, root, top=f"Class{c}")
        log(f"  {rel}: SHA-256 and published MD5 verified; {res['written']} written, "
            f"{res['unchanged']} identical, {res['skipped']} Thumbs.db skipped")
        got = res.pop("members")
        kinds = {"image": 0, "mask": 0, "labels": 0}
        for k in got:
            parts = k.split("/")
            if len(parts) == 3 and parts[1] in ("Train", "Test") and parts[2].endswith(".PNG"):
                kinds["image"] += 1
            elif len(parts) == 4 and parts[1] in ("Train", "Test") and parts[2] == "Label":
                if parts[3] == "Labels.txt":
                    kinds["labels"] += 1
                elif parts[3].endswith("_label.PNG"):
                    kinds["mask"] += 1
                else:
                    raise ArchiveError(f"Class{c}: unexpected member {k}")
            else:
                raise ArchiveError(f"Class{c}: unexpected member {k}")
        if kinds["image"] != DAGM_EXPECTED_FILES[c] or kinds["labels"] != 2:
            raise ArchiveError(f"Class{c}: {kinds} (expected {DAGM_EXPECTED_FILES[c]} images, 2 Labels.txt)")
        members.update(got)
        steps.append({**info, **res})
    status = _check_manifest(Path(cfg.dagm.manifest), members, write_manifests, log)
    disk = verify_tree(root, read_sha256_list(cfg.dagm.manifest),
                       scan_suffixes=(".png", ".txt", ".bmp", ".jpg"), ignore_dirs=("zips",))
    _raise_on_tree(disk, root)
    log(f"  {root}: all {disk['n_expected']} files match the manifest, no unexpected files")
    return {"archives": steps, "manifest": status, "files": len(members)}


def _raise_on_tree(rep: Dict, root: Path) -> None:
    bad = {k: rep[k] for k in ("missing", "modified", "unexpected") if rep[k]}
    if bad:
        raise ArchiveError(f"{root}: " + "; ".join(f"{k} {len(v)} (e.g. {v[:3]})" for k, v in bad.items()))
