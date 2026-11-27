"""The Phase 4 extractor must refuse every unsafe or inconsistent input."""

import hashlib
import stat
import zipfile

import pytest

from idefect.data.archives import (ArchiveError, read_sha256_list, safe_extract, verify_archive,
                                   verify_tree, write_sha256_list)


def _zip(path, members, symlink=None):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "/etc/passwd")
    return path


def test_extracts_verifies_and_skips_thumbs(tmp_path):
    z = _zip(tmp_path / "a.zip", {"Class1/Train/0001.PNG": b"img", "Class1/Thumbs.db": b"x"})
    res = safe_extract(z, tmp_path / "out", top="Class1")
    assert res["written"] == 1 and res["skipped"] == 1
    assert (tmp_path / "out/Class1/Train/0001.PNG").read_bytes() == b"img"
    assert not (tmp_path / "out/Class1/Thumbs.db").exists()
    assert res["members"] == {"Class1/Train/0001.PNG": hashlib.sha256(b"img").hexdigest()}
    again = safe_extract(z, tmp_path / "out", top="Class1")      # idempotent
    assert again["written"] == 0 and again["unchanged"] == 1


@pytest.mark.parametrize("name", ["../evil.txt", "/abs.txt", "Class1/../../x", "Class1\\x",
                                  "C:/x", "Other/x.PNG"])
def test_unsafe_names_rejected_before_writing(tmp_path, name):
    z = _zip(tmp_path / "a.zip", {"Class1/ok.PNG": b"ok", name: b"bad"})
    with pytest.raises(ArchiveError):
        safe_extract(z, tmp_path / "out", top="Class1")
    assert not (tmp_path / "out/Class1/ok.PNG").exists()           # nothing written at all


def test_symlink_member_rejected(tmp_path):
    z = _zip(tmp_path / "a.zip", {"Class1/ok.PNG": b"ok"}, symlink="Class1/link")
    with pytest.raises(ArchiveError, match="symlink"):
        safe_extract(z, tmp_path / "out", top="Class1")


def test_case_collision_rejected(tmp_path):
    z = _zip(tmp_path / "a.zip", {"Class1/a.PNG": b"1", "Class1/A.png": b"2"})
    with pytest.raises(ArchiveError, match="duplicate"):
        safe_extract(z, tmp_path / "out", top="Class1")


def test_modified_file_is_never_overwritten(tmp_path):
    z = _zip(tmp_path / "a.zip", {"Class1/0001.PNG": b"original"})
    safe_extract(z, tmp_path / "out", top="Class1")
    f = tmp_path / "out/Class1/0001.PNG"
    f.write_bytes(b"edited by a viewer")
    with pytest.raises(ArchiveError, match="differs"):
        safe_extract(z, tmp_path / "out", top="Class1")
    assert f.read_bytes() == b"edited by a viewer"                 # evidence kept


def test_corrupt_member_detected(tmp_path):
    z = _zip(tmp_path / "a.zip", {"Class1/0001.PNG": b"A" * 5000})
    raw = bytearray(z.read_bytes())
    with zipfile.ZipFile(z) as zf:
        info = zf.infolist()[0]
    data_start = info.header_offset + 30 + len(info.filename.encode())
    raw[data_start + 2] ^= 0xFF                                     # flip bits in the compressed data
    z.write_bytes(bytes(raw))
    with pytest.raises(Exception):                                  # BadZipFile, zlib.error or ArchiveError
        safe_extract(z, tmp_path / "out", top="Class1")
    assert not (tmp_path / "out/Class1/0001.PNG").exists()


def test_verify_archive_checks_every_digest(tmp_path):
    f = tmp_path / "x.zip"
    f.write_bytes(b"abc")
    sha, md5 = hashlib.sha256(b"abc").hexdigest(), hashlib.md5(b"abc").hexdigest()
    assert verify_archive(f, sha, size=3, md5=md5)["md5_checked"]
    for kwargs in ({"sha256": "0" * 64}, {"sha256": sha, "size": 4}, {"sha256": sha, "md5": "0" * 32}):
        with pytest.raises(ArchiveError):
            verify_archive(f, **kwargs)


def test_manifest_roundtrip_and_tree_check(tmp_path):
    root = tmp_path / "ds"
    (root / "a").mkdir(parents=True)
    (root / "a/1.png").write_bytes(b"1")
    (root / "a/2.png").write_bytes(b"2")
    man = tmp_path / "m.sha256"
    write_sha256_list({"a/1.png": hashlib.sha256(b"1").hexdigest(),
                       "a/2.png": hashlib.sha256(b"2").hexdigest()}, man)
    expected = read_sha256_list(man)
    assert verify_tree(root, expected, (".png",))["modified"] == []
    (root / "a/2.png").write_bytes(b"changed")
    (root / "a/3.PNG").write_bytes(b"3")
    (root / "a/1.png").unlink()
    rep = verify_tree(root, expected, (".png",))
    assert rep["missing"] == ["a/1.png"] and rep["modified"] == ["a/2.png"]
    assert rep["unexpected"] == ["a/3.PNG"]


def test_malformed_manifest_rejected(tmp_path):
    man = tmp_path / "m.sha256"
    man.write_text("nothex  a.png\n")
    with pytest.raises(ArchiveError):
        read_sha256_list(man)
    man.write_text(f"{'a' * 64}  a.png\n{'b' * 64}  a.png\n")
    with pytest.raises(ArchiveError, match="duplicate"):
        read_sha256_list(man)
