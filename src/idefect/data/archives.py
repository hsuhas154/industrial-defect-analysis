"""Verified, fail-closed extraction of the Phase 4 dataset archives.

Every step can only stop the run, never silently continue:

1. Each archive is checked against the SHA-256 and byte size in
   ``data/phase4_archives.sha256`` (written when the downloads were first
   verified) and, for DAGM, against the MD5 that Zenodo publishes for record
   10.5281/zenodo.8086136. A mismatch stops the run.
2. Every member name is checked before anything is written: no absolute paths,
   no ``..``, no backslashes, no drive letters, no symlinks, no encrypted
   members, no duplicate names, and every name must start with the expected
   top-level folder. One bad member rejects the whole archive.
3. Each file is written to a temporary name, its size and CRC-32 are compared
   with the archive's record, and only then is it renamed into place.
4. A file that already exists is never overwritten: if its bytes equal the
   archive member it is left alone, otherwise the run stops and names it (an
   edited or re-saved image is a finding, not something to paper over).
5. ``Thumbs.db`` (Windows thumbnail caches shipped inside the DAGM zips) is
   skipped; nothing else is.

The integrity manifest of the extracted files is computed from the archive
members themselves, and then the files on disk are verified against it, so the
manifest and the disk are two independent routes to the same bytes.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SKIP_NAMES = frozenset({"thumbs.db"})

# Published MD5 checksums (Zenodo record 10.5281/zenodo.8086136, CC BY 4.0).
# The later record 10.5281/zenodo.12750201 lists the same ten files with the
# same checksums. KolektorSDD publishes no checksum; its archives are pinned by
# the SHA-256 recorded at download time only.
DAGM_PUBLISHED_MD5 = {
    "Class1.zip": "9ed7f0480aea9bd0a83b429a17e3e750",
    "Class2.zip": "024f1f6c6d11d72ee12f19dcc98aa3d8",
    "Class3.zip": "57a337e93e38c71e01d15770e252f46d",
    "Class4.zip": "d6852ae3667db59e6a4d586d98633540",
    "Class5.zip": "463a932d87a5c0e02ee10cdb3265c318",
    "Class6.zip": "40ba70f232fe1cb891a46c1b77dc4940",
    "Class7.zip": "67571381c8aa5f82d5180e6e8838f735",
    "Class8.zip": "d02a892a8b6180893c8ea018848606cb",
    "Class9.zip": "083d8da49d46614202a332f9e9dd1bca",
    "Class10.zip": "2743ebc79cb91551b51599a67f0e7528",
}

_CHUNK = 1 << 20


class ArchiveError(RuntimeError):
    """An archive or an extracted file failed a check; nothing was guessed."""


def file_digests(path: str | Path) -> Tuple[str, str, int]:
    """(sha256, md5, size) of a file, read in chunks."""
    sha, md5, n = hashlib.sha256(), hashlib.md5(), 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            sha.update(block)
            md5.update(block)
            n += len(block)
    return sha.hexdigest(), md5.hexdigest(), n


def read_sha256_list(path: str | Path) -> Dict[str, str]:
    """Parse a ``sha256sum`` file into {relative path: digest}; strict format."""
    out: Dict[str, str] = {}
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64 or any(c not in "0123456789abcdef" for c in parts[0]):
            raise ArchiveError(f"{path}:{lineno}: not a sha256sum line: {line!r}")
        name = parts[1].strip().lstrip("*")
        if name in out:
            raise ArchiveError(f"{path}:{lineno}: duplicate entry {name}")
        out[name] = parts[0]
    return out


def write_sha256_list(entries: Dict[str, str], path: str | Path) -> None:
    lines = [f"{digest}  {name}" for name, digest in sorted(entries.items())]
    Path(path).write_text("\n".join(lines) + "\n")


def verify_archive(path: str | Path, sha256: str, size: Optional[int] = None,
                   md5: Optional[str] = None) -> Dict:
    """Check one archive against known digests; raises on any mismatch."""
    path = Path(path)
    if not path.is_file():
        raise ArchiveError(f"archive not found: {path}")
    got_sha, got_md5, got_size = file_digests(path)
    if got_sha != sha256:
        raise ArchiveError(f"{path}: SHA-256 {got_sha} differs from the recorded {sha256}")
    if size is not None and got_size != size:
        raise ArchiveError(f"{path}: size {got_size} differs from the recorded {size}")
    if md5 is not None and got_md5 != md5:
        raise ArchiveError(f"{path}: MD5 {got_md5} differs from the published {md5}")
    return {"archive": str(path), "sha256": got_sha, "md5": got_md5, "size": got_size,
            "md5_checked": md5 is not None}


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return info.create_system == 3 and stat.S_ISLNK(info.external_attr >> 16)


def check_member_name(name: str, top: str) -> PurePosixPath:
    """Validate one member name; returns it as a relative POSIX path.

    ``top`` is a regular expression the first path component must match in
    full (empty: anything)."""
    if "\\" in name or "\x00" in name or name.startswith("/") or ":" in name:
        raise ArchiveError(f"unsafe member name {name!r}")
    p = PurePosixPath(name)
    if p.is_absolute() or any(part in ("..", ".", "") for part in p.parts):
        raise ArchiveError(f"unsafe member name {name!r}")
    if top and not re.fullmatch(top, p.parts[0]):
        raise ArchiveError(f"member {name!r} is outside the expected folder {top!r}")
    return p


def plan_members(zf: zipfile.ZipFile, top: str = "") -> List[Tuple[zipfile.ZipInfo, PurePosixPath]]:
    """Validate every member first; return the regular files to extract."""
    seen, out = set(), []
    for info in zf.infolist():
        rel = check_member_name(info.filename.rstrip("/") if info.is_dir() else info.filename, top)
        key = str(rel).lower()  # also refuse names that collide on a case-insensitive disk
        if key in seen:
            raise ArchiveError(f"duplicate member {info.filename!r}")
        seen.add(key)
        if info.flag_bits & 0x1:
            raise ArchiveError(f"encrypted member {info.filename!r}")
        if _is_symlink(info):
            raise ArchiveError(f"symlink member {info.filename!r}")
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise ArchiveError(f"unexpected compression {info.compress_type} for {info.filename!r}")
        if info.is_dir() or rel.name.lower() in SKIP_NAMES:
            continue
        out.append((info, rel))
    return out


def _member_bytes_digest(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> Tuple[str, int, int]:
    """(sha256, crc32, size) of a member's decompressed bytes."""
    sha, crc, n = hashlib.sha256(), 0, 0
    with zf.open(info) as fh:  # zipfile itself raises BadZipFile on a CRC mismatch at EOF
        for block in iter(lambda: fh.read(_CHUNK), b""):
            sha.update(block)
            crc = zlib.crc32(block, crc)
            n += len(block)
    return sha.hexdigest(), crc & 0xFFFFFFFF, n


def _disk_digest(path: Path) -> Tuple[str, int, int]:
    sha, crc, n = hashlib.sha256(), 0, 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            sha.update(block)
            crc = zlib.crc32(block, crc)
            n += len(block)
    return sha.hexdigest(), crc & 0xFFFFFFFF, n


def safe_extract(archive: str | Path, dest: str | Path, top: str = "",
                 strip_top: bool = False, prefix: str = "") -> Dict:
    """Extract ``archive`` into ``dest`` with every check described above.

    ``top`` is a regular expression for the first path component of every
    member (empty: no constraint). ``strip_top`` drops that component;
    ``prefix`` is prepended to the written path (for example ``splits``).
    Returns {"written", "unchanged", "skipped", "members": {relpath: sha256}},
    where relpath is relative to ``dest``.
    """
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    dest_real = dest.resolve()
    written = unchanged = 0
    members: Dict[str, str] = {}
    with zipfile.ZipFile(archive) as zf:
        plan = plan_members(zf, top)
        skipped = sum(1 for i in zf.infolist() if not i.is_dir()) - len(plan)
        for info, rel in plan:
            parts = rel.parts[1:] if strip_top else rel.parts
            out_rel = PurePosixPath(prefix, *parts) if prefix else PurePosixPath(*parts)
            target = dest / out_rel
            if dest_real not in target.resolve().parents:
                raise ArchiveError(f"{info.filename!r} would land outside {dest}")
            sha, crc, size = _member_bytes_digest(zf, info)
            if crc != info.CRC or size != info.file_size:
                raise ArchiveError(f"{archive}:{info.filename}: CRC or size differs from the archive record")
            key = out_rel.as_posix()
            if key in members:
                raise ArchiveError(f"two members map to {key}")
            members[key] = sha
            if target.is_symlink():
                raise ArchiveError(f"{target} is a symlink; refusing to touch it")
            if target.exists():
                if _disk_digest(target)[0] != sha:
                    raise ArchiveError(
                        f"{target} exists and differs from {archive.name}:{info.filename}. "
                        "It was changed after extraction; restore it (delete it and rerun) "
                        "after finding out what modified it.")
                unchanged += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".partial")
            if tmp.is_file() and not tmp.is_symlink():
                tmp.unlink()  # left behind by an interrupted run; never a dataset file
            try:
                with zf.open(info) as src, open(tmp, "xb") as dst:
                    for block in iter(lambda: src.read(_CHUNK), b""):
                        dst.write(block)
                    dst.flush()
                    os.fsync(dst.fileno())
                d_sha, d_crc, d_size = _disk_digest(tmp)
                if (d_sha, d_crc, d_size) != (sha, info.CRC, info.file_size):
                    raise ArchiveError(f"{tmp}: written bytes differ from the archive member")
                os.replace(tmp, target)
            finally:
                if tmp.exists():
                    tmp.unlink()
            written += 1
    return {"archive": str(archive), "written": written, "unchanged": unchanged,
            "skipped": skipped, "members": members}


def verify_tree(root: str | Path, expected: Dict[str, str],
                scan_suffixes: Iterable[str] = (), ignore_dirs: Sequence[str] = ()) -> Dict:
    """Verify files under ``root`` against {relpath: sha256}.

    Reports missing and modified entries, and unexpected files: any file under
    ``root`` with a suffix in ``scan_suffixes`` (case-insensitive) that is not
    in ``expected``, outside ``ignore_dirs`` (top-level folder names).
    """
    root = Path(root)
    missing, modified = [], []
    for rel, digest in sorted(expected.items()):
        f = root / rel
        if not f.is_file():
            missing.append(rel)
        elif _disk_digest(f)[0] != digest:
            modified.append(rel)
    suffixes = {s.lower() for s in scan_suffixes}
    unexpected = []
    if suffixes and root.is_dir():
        for f in root.rglob("*"):
            rel = f.relative_to(root)
            if rel.parts[0] in ignore_dirs or not f.is_file():
                continue
            if f.suffix.lower() in suffixes and rel.as_posix() not in expected:
                unexpected.append(rel.as_posix())
    return {"checked": True, "n_expected": len(expected), "missing": missing,
            "modified": modified, "unexpected": sorted(unexpected)}
