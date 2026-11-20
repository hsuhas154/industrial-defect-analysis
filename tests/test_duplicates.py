"""Near-duplicate detector: planted copies must be grouped, distinct images not.

The synthetic surfaces mimic steel: band-limited random grain plus optional
strong structures. The hardest negative is two different surfaces that share
an identical bright line (like two crops along one long scratch): their raw
correlation is high, yet they share no pixels and no background grain.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")

from idefect.config import DuplicateConfig  # noqa: E402
from idefect.data.datasets import make_splits  # noqa: E402
from idefect.data.duplicates import (  # noqa: E402
    detect_duplicates,
    dihedral,
    match_against,
    overlap_views,
)

S = 128


def _grain(rng, size=S, base=120):
    noise = rng.normal(0, 1, (size, size)).astype(np.float32)
    g = cv2.GaussianBlur(noise, (0, 0), 1.2)
    g = g / g.std() * 12 + base
    shade = np.linspace(-10, 10, size, dtype=np.float32)[None, :]  # illumination gradient
    return np.clip(g + shade, 0, 255)


def _with_line(img, x0, width=10):
    out = img.copy()
    out[:, x0:x0 + width] = np.clip(out[:, x0:x0 + width] + 90, 0, 255)
    return out


def _u8(x):
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    root = tmp_path_factory.mktemp("dups")
    rng = np.random.default_rng(0)
    big = _grain(rng, size=2 * S)  # a larger surface to cut overlapping crops from
    imgs = {}
    for i in range(10):
        imgs[f"distinct_{i}.png"] = _u8(_grain(rng))
    src = _u8(_grain(rng))
    imgs["src.png"] = src
    imgs["copy_exact.png"] = src.copy()
    imgs["copy_rot90.png"] = dihedral(src, 1)
    imgs["copy_flip.png"] = dihedral(src, 4)
    imgs["copy_bright.png"] = _u8(src.astype(np.float32) * 0.8 + 30)
    imgs["copy_jpeg.png"] = cv2.imdecode(cv2.imencode(".jpg", src, [cv2.IMWRITE_JPEG_QUALITY, 75])[1], 0)
    m = cv2.getRotationMatrix2D(((S - 1) / 2, (S - 1) / 2), 0.4, 1.0)
    imgs["copy_resampled.png"] = cv2.warpAffine(src, m, (S, S), borderMode=cv2.BORDER_REFLECT)
    imgs["crop_a.png"] = _u8(big[:S, :S])
    imgs["crop_b.png"] = _u8(big[40:40 + S, 50:50 + S])          # about 42 percent overlap
    imgs["crop_c_rot.png"] = dihedral(_u8(big[64:64 + S, 0:S]), 3)  # 50 percent overlap, rotated
    line_a = _u8(_with_line(_grain(rng), 50))
    line_b = _u8(_with_line(_grain(rng), 50))
    imgs["line_a.png"], imgs["line_b.png"] = line_a, line_b
    for name, im in imgs.items():
        cv2.imwrite(str(root / name), im)
    paths = sorted(str(root / n) for n in imgs)
    cfg = DuplicateConfig(cache_dir="", review_file="", top_k=6)
    return root, paths, detect_duplicates(paths, cfg, use_cache=False), cfg


def _group_of(report, name):
    for g in report.groups:
        if any(Path(p).name == name for p in g):
            return {Path(p).name for p in g}
    return {name}


def test_all_planted_copies_are_grouped(planted):
    _, _, rep, _ = planted
    g = _group_of(rep, "src.png")
    for name in ["copy_exact.png", "copy_rot90.png", "copy_flip.png", "copy_bright.png",
                 "copy_jpeg.png", "copy_resampled.png"]:
        assert name in g, f"{name} not grouped with src.png; groups={rep.groups}"


def test_overlapping_crops_are_grouped(planted):
    _, _, rep, _ = planted
    g = _group_of(rep, "crop_a.png")
    assert {"crop_b.png", "crop_c_rot.png"} <= g


def test_distinct_images_stay_apart(planted):
    _, _, rep, _ = planted
    grouped = {Path(p).name for g in rep.groups for p in g}
    assert not any(n.startswith("distinct_") for n in grouped), rep.groups
    # Same bright line, different surface: must not be passed off as the same image.
    assert "line_b.png" not in _group_of(rep, "line_a.png")


def test_groups_are_disjoint_and_review_is_honoured(planted, tmp_path):
    root, paths, rep, cfg = planted
    seen = set()
    for g in rep.groups:
        assert not (set(g) & seen)
        seen |= set(g)
    review = tmp_path / "review.yaml"
    review.write_text("same:\n  - [distinct_0.png, distinct_1.png]\ndistinct: []\n")
    cfg2 = DuplicateConfig(cache_dir="", review_file=str(review), top_k=6)
    rep2 = detect_duplicates(paths, cfg2, use_cache=False)
    assert "distinct_1.png" in _group_of(rep2, "distinct_0.png")


def test_match_against_finds_resized_copy(planted):
    root, paths, _, cfg = planted
    src = cv2.imread(str(root / "src.png"), 0)
    query = cv2.resize(src, (2 * S + 7, 2 * S + 7), interpolation=cv2.INTER_CUBIC)
    refs = [p for p in paths if Path(p).name.startswith(("distinct_", "src"))]
    top = match_against(query, refs, cfg, top_n=1)
    assert top and Path(top[0].b).name == "src.png" and top[0].kind == "same"


def test_match_against_finds_rotated_crop_at_native_scale(planted):
    # A 100x100 crop of crop_a, rotated: resizing it to 128x128 would destroy the
    # pixel identity, so only the native-scale search can find it.
    root, paths, _, cfg = planted
    crop_a = cv2.imread(str(root / "crop_a.png"), 0)
    query = dihedral(crop_a[10:110, 20:120], 1)
    top = match_against(query, paths, cfg, top_n=3)
    assert top and Path(top[0].b).name == "crop_a.png", top
    assert top[0].kind == "same" and top[0].evidence == "pixel-identical"
    assert top[0].overlap_px == 100 * 100
    # crop_a rotated by 90 degrees is what contains the query
    assert top[0].transform == "rot90"
    # neighbouring crops share fewer pixels and rank below the source
    assert all(p.overlap_px < 100 * 100 for p in top[1:] if p.evidence == "pixel-identical")


def test_inverse_dihedral_undoes_every_transform():
    from idefect.data.duplicates import _inverse_dihedral

    x = np.arange(12).reshape(3, 4)
    for k in range(8):
        assert np.array_equal(dihedral(dihedral(x, k), _inverse_dihedral(k)), x), k


def test_cache_only_never_runs_the_detection(planted, tmp_path):
    from idefect.data.duplicates import DuplicateCacheMissing

    _, paths, _, _ = planted
    cfg = DuplicateConfig(cache_dir=str(tmp_path / "empty_cache"), review_file="", top_k=6)
    with pytest.raises(DuplicateCacheMissing):
        detect_duplicates(paths, cfg, cache_only=True)
    assert not (tmp_path / "empty_cache").exists()


def test_overlap_views_general_shapes():
    a = np.arange(20).reshape(4, 5)
    b = np.arange(12).reshape(3, 4)
    va, vb = overlap_views(a, b, 1, -1)
    assert va.shape == vb.shape and va.size > 0


def test_split_keeps_cross_class_group_together():
    samples = [(f"a{i}", 0) for i in range(20)] + [(f"b{i}", 1) for i in range(20)]
    groups = [["a0", "b0", "a1"], ["b5", "b6", "b7", "b8", "b9", "b10"]]
    for seed in range(15):
        sp = make_splits(samples, ["A", "B"], 0.6, 0.2, 0.2, seed=seed, groups=groups)
        where = {p: n for n, part in (("tr", sp.train), ("va", sp.val), ("te", sp.test)) for p, _ in part}
        for g in groups:
            assert len({where[p] for p in g}) == 1
        assert len(sp.train) + len(sp.val) + len(sp.test) == 40
        # proportions stay close despite a 6-image group
        assert abs(len(sp.train) - 24) <= 6


def test_leakage_rate_counts_test_images_sharing_a_group_with_train():
    from idefect.data.datasets import DatasetSplit
    from idefect.training.studies import leakage_rate

    split = DatasetSplit(train=[("a", 0), ("b", 0)], val=[("c", 0)], test=[("d", 1), ("e", 1)],
                         class_names=["x", "y"])
    rate = leakage_rate(split, [["a", "d"], ["c", "e"]])
    assert rate["test"]["leaked"] == 1 and rate["val"]["leaked"] == 0


def test_pair_gallery_renders(planted, tmp_path):
    from idefect.visualization.duplicates import plot_pairs

    _, _, rep, _ = planted
    same = [p for p in rep.pairs if p.kind == "same"]
    out = plot_pairs(same, tmp_path / "pairs.png", "planted copies")
    assert out is not None and out.exists()
