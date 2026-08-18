"""Tests for dataset discovery and deterministic stratified splitting."""

from idefect import NEU_CLASSES
from idefect.data.datasets import load_samples, make_splits


def test_load_samples_finds_all_classes(dummy_dataset):
    samples, class_names = load_samples(dummy_dataset)
    assert sorted(class_names) == sorted(NEU_CLASSES)
    assert len(samples) == len(NEU_CLASSES) * 8
    # labels are valid indices
    assert all(0 <= label < len(class_names) for _, label in samples)


def test_make_splits_is_deterministic(dummy_dataset):
    samples, class_names = load_samples(dummy_dataset)
    kw = dict(train_frac=0.5, val_frac=0.25, test_frac=0.25, seed=123)
    s1 = make_splits(samples, class_names, **kw)
    s2 = make_splits(samples, class_names, **kw)
    assert [p for p, _ in s1.train] == [p for p, _ in s2.train]
    assert [p for p, _ in s1.test] == [p for p, _ in s2.test]


def test_splits_are_disjoint_and_cover_everything(dummy_dataset):
    samples, class_names = load_samples(dummy_dataset)
    split = make_splits(samples, class_names, 0.6, 0.2, 0.2, seed=7)
    train = {p for p, _ in split.train}
    val = {p for p, _ in split.val}
    test = {p for p, _ in split.test}
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train | val | test) == len(samples)


def test_splits_are_stratified(dummy_dataset):
    samples, class_names = load_samples(dummy_dataset)
    split = make_splits(samples, class_names, 0.5, 0.25, 0.25, seed=42)
    # With 8 per class and a 50% train fraction, every class has 4 train samples.
    counts = {i: 0 for i in range(len(class_names))}
    for _, label in split.train:
        counts[label] += 1
    assert all(c == 4 for c in counts.values())
