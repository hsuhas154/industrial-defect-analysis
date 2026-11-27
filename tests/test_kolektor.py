"""KolektorSDD loader, safe split decoding and leakage-safe folds (synthetic data)."""

import os
import pickle

import cv2
import numpy as np
import pytest
from PIL import Image

from idefect.data.kolektor import (DatasetError, OfficialFolds, PartRelation, UnsafePickle, check_folds,
                                   corrected_folds, crossing_relations, decode_split_pickle,
                                   load_kolektorsdd, load_official_folds, part_relations, read_mask)


# --------------------------------------------------------------------------
# split file decoding
# --------------------------------------------------------------------------
@pytest.mark.parametrize("protocol", [0, 1, 2, 4])
def test_decoder_matches_pickle_on_plain_data(protocol):
    obj = [[["kos01", "kos02"], ["kos03"]], [["kos03"], ["kos01", "kos02"]], ["kos01", "kos02", "kos03"]]
    data = pickle.dumps(obj, protocol=protocol)
    assert decode_split_pickle(data) == pickle.loads(data)


class _Evil:
    def __reduce__(self):
        return (os.getcwd, ())


@pytest.mark.parametrize("protocol", [0, 2, 4])
def test_decoder_refuses_code(protocol):
    data = pickle.dumps([["kos01"], _Evil()], protocol=protocol)
    with pytest.raises(UnsafePickle):
        decode_split_pickle(data)


def test_decoder_refuses_other_objects():
    with pytest.raises(UnsafePickle):
        decode_split_pickle(pickle.dumps({"a": 1}, protocol=0))
    with pytest.raises(UnsafePickle):
        decode_split_pickle(b"(lp0\nS'kos01'\np1\na")       # no STOP


def _write_split(path, train, test, parts):
    path.write_bytes(pickle.dumps([train, test, parts], protocol=0))
    return path


def test_official_folds_structure_checked(tmp_path):
    parts = ["kos01", "kos02", "kos03", "kos04"]
    ok = _write_split(tmp_path / "ok.pyb", [["kos03", "kos04"], ["kos01", "kos02"]],
                      [["kos01", "kos02"], ["kos03", "kos04"]], parts)
    f = load_official_folds(ok)
    assert f.test == [["kos01", "kos02"], ["kos03", "kos04"]]
    bad = _write_split(tmp_path / "overlap.pyb", [["kos03", "kos04"], ["kos02"]],
                       [["kos01", "kos02"], ["kos01", "kos03", "kos04"]], parts)
    with pytest.raises(DatasetError):
        load_official_folds(bad)
    bad2 = _write_split(tmp_path / "both.pyb", [["kos01", "kos03", "kos04"], ["kos01", "kos02"]],
                        [["kos01", "kos02"], ["kos03", "kos04"]], parts)
    with pytest.raises(DatasetError):
        load_official_folds(bad2)


# --------------------------------------------------------------------------
# masks and loader
# --------------------------------------------------------------------------
def _palette_mask(path, shape, box, index=1, palette=None):
    m = np.zeros(shape, np.uint8)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = index
    im = Image.fromarray(m, mode="P")
    pal = palette or [0, 0, 0, 255, 255, 255]
    im.putpalette(pal + [0] * (768 - len(pal)))
    im.save(path)


def test_palette_mask_read_as_above_zero(tmp_path):
    # kos20/Part4 style: index 1 = grey 85, index 2 = white; both are defect
    m = np.zeros((40, 30), np.uint8)
    m[5:10, 5:10] = 1
    m[20:30, 10:20] = 2
    im = Image.fromarray(m, mode="P")
    im.putpalette([0, 0, 0, 85, 85, 85, 255, 255, 255] + [0] * 759)
    im.save(tmp_path / "m.bmp")
    got = read_mask(tmp_path / "m.bmp")
    assert got.sum() == 25 + 100
    assert np.array_equal(got, m > 0)


def _texture(seed, shape=(60, 40)):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur((rng.random(shape) * 255).astype(np.uint8), (0, 0), 1.0)


def _make_tree(root, n_parts=6, per_part=3, copy=("kos01", "kos05")):
    imgs = {}
    for p in range(1, n_parts + 1):
        part = f"kos{p:02d}"
        (root / part).mkdir(parents=True)
        for i in range(per_part):
            img = _texture(100 * p + i)
            imgs[(part, i)] = img
            cv2.imwrite(str(root / part / f"Part{i}.jpg"), img)
            if i == 0:
                _palette_mask(root / part / f"Part{i}_label.bmp", img.shape, (10, 20, 5, 15))
            else:
                cv2.imwrite(str(root / part / f"Part{i}_label.bmp"), np.zeros(img.shape, np.uint8))
    if copy:
        src, dst = copy
        for f in (root / src).iterdir():
            (root / dst / f.name).write_bytes(f.read_bytes())
    return root


def test_loader_counts_and_strictness(tmp_path):
    root = _make_tree(tmp_path / "ksdd")
    s = load_kolektorsdd(root)
    assert len(s) == 18 and sum(x.defective for x in s) == 6
    assert all(x.defect_px == 100 for x in s if x.defective)
    (root / "kos02" / "notes.txt").write_text("x")
    with pytest.raises(DatasetError, match="unexpected file"):
        load_kolektorsdd(root)
    (root / "kos02" / "notes.txt").unlink()
    (root / "kos03" / "Part1_label.bmp").unlink()
    with pytest.raises(DatasetError, match="differ"):
        load_kolektorsdd(root)


# --------------------------------------------------------------------------
# relations and folds
# --------------------------------------------------------------------------
def test_relations_find_copy_and_related(tmp_path):
    s = load_kolektorsdd(_make_tree(tmp_path / "ksdd"))
    rel = part_relations(s, near_pairs=[("kos02/Part1", "kos06/Part2")])
    kinds = {tuple(r.parts): r.kind for r in rel}
    assert kinds == {("kos01", "kos05"): "copy", ("kos02", "kos06"): "related"}
    with pytest.raises(DatasetError):
        part_relations(s, near_pairs=[("kos02/Part1", "kos99/Part0")])


def _official():
    parts = [f"kos{i:02d}" for i in range(1, 7)]
    test = [["kos01", "kos02"], ["kos03", "kos05"], ["kos04", "kos06"]]
    return OfficialFolds(train=[sorted(set(parts) - set(t)) for t in test], test=test, parts=parts)


REL = [PartRelation(["kos01", "kos05"], "copy", ["identical-file"]),
       PartRelation(["kos02", "kos06"], "related", ["near-duplicate"])]


@pytest.mark.parametrize("policy", ["drop_copies", "keep_together"])
def test_corrected_folds_never_cross(policy):
    off = _official()
    assert any(crossing_relations(tr, te, REL) for tr, te in zip(off.train, off.test))  # the leak exists
    f = corrected_folds(off, REL, policy, val_frac=0.3, seed=1)
    kept = sorted(set(off.parts) - set(f.dropped))
    for tr, va, te in zip(f.train, f.val, f.test):
        assert not crossing_relations(tr + va, te, REL)
        assert not crossing_relations(tr, va, REL)
        assert sorted(tr + va + te) == kept
    assert sorted(p for t in f.test for p in t) == kept
    if policy == "drop_copies":
        assert f.dropped == ["kos05"]
    else:
        assert f.dropped == [] and any({"kos01", "kos05"} <= set(t) for t in f.test)
    assert any({"kos02", "kos06"} <= set(t) for t in f.test)       # related parts kept together


def test_folds_deterministic_and_seed_dependent():
    off = _official()
    a = corrected_folds(off, REL, "drop_copies", 0.3, 42)
    b = corrected_folds(off, REL, "drop_copies", 0.3, 42)
    assert a == b
    vals = {tuple(map(tuple, corrected_folds(off, REL, "drop_copies", 0.3, s).val)) for s in range(10)}
    assert len(vals) > 1


def test_check_folds_catches_a_crossing():
    off = _official()
    f = corrected_folds(off, REL, "drop_copies", 0.3, 42)
    k = next(i for i, t in enumerate(f.test) if "kos02" in t)
    f.test[k].remove("kos06")
    f.train[k].append("kos06")
    with pytest.raises(DatasetError):
        check_folds(f, sorted(set(off.parts) - set(f.dropped)), REL)
