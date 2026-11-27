"""DAGM label parsing, strict loading and the validation split (synthetic data)."""

from collections import Counter

import cv2
import numpy as np
import pytest

from idefect.data.dagm import load_dagm, parse_labels, validation_split
from idefect.data.kolektor import DatasetError


def _labels(path, rows):
    lines = ["1"]
    for rid, flag in rows:
        lab = f"{rid}_label.PNG" if flag else "0"
        lines.append(f"{rid}\t{flag}\t{rid}.PNG\t0\t{lab}\t")
    path.write_text("\r\n".join(lines) + "\r\n")


def _make_class(root, c, n_train=20, n_test=10, defect_every=4):
    rid = 0
    for split, n in (("Train", n_train), ("Test", n_test)):
        d = root / f"Class{c}" / split
        (d / "Label").mkdir(parents=True)
        rows = []
        for _ in range(n):
            rid += 1
            name = f"{rid:04d}"
            flag = int(rid % defect_every == 0)
            rows.append((name, flag))
            cv2.imwrite(str(d / f"{name}.PNG"), np.full((16, 16), rid % 255, np.uint8))
            if flag:
                m = np.zeros((16, 16), np.uint8)
                m[4:8, 4:8] = 255
                cv2.imwrite(str(d / "Label" / f"{name}_label.PNG"), m)
        _labels(d / "Label" / "Labels.txt", rows)
        (d / "Thumbs.db").write_bytes(b"windows cache")
    return root


def test_parse_labels_strict(tmp_path):
    p = tmp_path / "Labels.txt"
    _labels(p, [("0001", 0), ("0002", 1)])
    rows = parse_labels(p)
    assert [(r.name, r.defective, r.label) for r in rows] == [("0001.PNG", False, None),
                                                               ("0002.PNG", True, "0002_label.PNG")]
    for bad in ["0001\t1\t0001.PNG\t0\t0\t",            # defect flag without label
                "0001\t0\t0001.PNG\t0\t0001_label.PNG",   # label without defect flag
                "0001\t0\t0002.PNG\t0\t0",                # id and name disagree
                "0001\t2\t0001.PNG\t0\t0"]:               # bad flag
        p.write_text("1\r\n" + bad + "\r\n")
        with pytest.raises(DatasetError):
            parse_labels(p)
    p.write_text("0001\t0\t0001.PNG\t0\t0\r\n")             # missing header
    with pytest.raises(DatasetError):
        parse_labels(p)


def test_loader_cross_checks_folder_and_labels(tmp_path):
    root = _make_class(tmp_path / "dagm", 1)
    s = load_dagm(root, [1])
    assert len(s) == 30 and sum(x.defective for x in s) == 7      # Thumbs.db ignored
    (root / "Class1/Train/0001.PNG").unlink()
    with pytest.raises(DatasetError):
        load_dagm(root, [1])


def test_loader_rejects_orphan_mask(tmp_path):
    root = _make_class(tmp_path / "dagm", 1)
    cv2.imwrite(str(root / "Class1/Test/Label/0021_label.PNG"), np.zeros((16, 16), np.uint8))
    with pytest.raises(DatasetError, match="mask"):
        load_dagm(root, [1])


def test_validation_split_disjoint_stratified_deterministic(tmp_path):
    root = _make_class(tmp_path / "dagm", 1, n_train=40, n_test=20)
    _make_class(root, 2, n_train=40, n_test=20)
    s = load_dagm(root, [1, 2])
    a = validation_split(s, 0.25, 7)
    assert a == validation_split(s, 0.25, 7)
    assert a != validation_split(s, 0.25, 8)
    assert all(a[x.id] == "test" for x in s if x.split == "Test")
    assert all(a[x.id] in ("train", "val") for x in s if x.split == "Train")
    cnt = Counter((x.cls, x.defective, a[x.id]) for x in s if x.split == "Train")
    for c in (1, 2):
        for flag in (False, True):
            n = cnt[(c, flag, "train")] + cnt[(c, flag, "val")]
            assert cnt[(c, flag, "val")] == round(0.25 * n)


def test_validation_split_keeps_groups_whole(tmp_path):
    s = load_dagm(_make_class(tmp_path / "dagm", 1, n_train=40, n_test=20), [1])
    group = ["Class1/Train/0001.PNG", "Class1/Train/0002.PNG", "Class1/Train/0003.PNG"]
    for seed in range(20):
        a = validation_split(s, 0.25, seed, [group])
        assert len({a[i] for i in group}) == 1
    with pytest.raises(DatasetError):
        validation_split(s, 0.25, 0, [["Class1/Train/0001.PNG", "Class1/Test/0041.PNG"]])
