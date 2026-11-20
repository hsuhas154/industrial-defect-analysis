"""Everything that is fitted sees training images only.

Spies on the real training entry points:

* classical branch: the hyperparameter search (scalers, classifiers and their
  cross-validation all live inside it) must receive exactly the training
  features, and none of the validation or test rows;
* CNN: every optimizer step runs in ``_train_epoch``, whose loader must hold
  exactly the training images, in every epoch;
* both: the input-check reference statistics stored with each model come from
  training images only.
"""

import numpy as np
import pytest

from idefect.config import Config


@pytest.fixture
def cfg(dummy_dataset, tmp_path) -> Config:
    c = Config()
    c.data.root = str(dummy_dataset)
    c.data.manifest = ""
    c.data.image_size = 32
    c.data.train_frac, c.data.val_frac, c.data.test_frac = 0.5, 0.25, 0.25
    c.hog.pixels_per_cell = [8, 8]
    c.classical_ml.feature_sets = ["lbp"]
    c.classical_ml.models = ["logreg"]
    c.classical_ml.cv_folds = 3
    c.classical_ml.n_jobs = 1
    c.cnn.epochs = 2
    c.cnn.batch_size = 8
    c.cnn.num_workers = 0
    c.cnn.base_channels = 8
    c.cnn.device = "cpu"
    c.cnn.seeds = [0]
    c.data.duplicates.cache_dir = str(tmp_path / "cache")
    c.data.duplicates.review_file = ""
    c.data.duplicates.top_k = 3
    c.output.results_dir = str(tmp_path / "results")
    c.output.figures_dir = str(tmp_path / "figures")
    c.output.logs_dir = str(tmp_path / "logs")
    return c


def test_classical_fit_sees_training_features_only(cfg, monkeypatch):
    import idefect.training.train_classical_ml as T
    from idefect.features.registry import extract_many

    seen = {"X": [], "y": [], "ref": []}
    real_tune, real_ref = T.tune, T.reference_stats

    def spy_tune(model, X, y, groups, c, verbose=True):
        seen["X"].append(np.array(X, copy=True))
        seen["y"].append(np.array(y, copy=True))
        return real_tune(model, X, y, groups, c, verbose=verbose)

    def spy_ref(paths):
        seen["ref"].append(list(paths))
        return real_ref(paths)

    monkeypatch.setattr(T, "tune", spy_tune)
    monkeypatch.setattr(T, "reference_stats", spy_ref)
    T.train_classical_ml(cfg)

    _, split, _, arrays, _ = T.prepare_data(cfg, verbose=False)
    F = {p: extract_many(arrays[p][0], "lbp", cfg, n_jobs=1) for p in ("train", "val", "test")}
    held_out = np.concatenate([F["val"], F["test"]])
    assert seen["X"], "the search was never called"
    for X, y in zip(seen["X"], seen["y"]):
        assert np.array_equal(X, F["train"]) and np.array_equal(y, arrays["train"][1])
        assert not any((X == row).all(axis=1).any() for row in held_out)
    train_paths = {p for p, _ in split.train}
    assert seen["ref"] and all(set(p) == train_paths for p in seen["ref"])


def test_cnn_optimizer_sees_training_images_only(cfg, monkeypatch):
    pytest.importorskip("torch")
    import idefect.training.train_cnn as C
    from idefect.data.datasets import build_split

    seen = {"epochs": [], "ref": []}
    real_epoch, real_ref = C._train_epoch, C.reference_stats

    def spy_epoch(model, loader, criterion, device, optimizer):
        seen["epochs"].append(list(loader.dataset.samples))
        return real_epoch(model, loader, criterion, device, optimizer)

    def spy_ref(paths):
        seen["ref"].append(list(paths))
        return real_ref(paths)

    monkeypatch.setattr(C, "_train_epoch", spy_epoch)
    monkeypatch.setattr(C, "reference_stats", spy_ref)
    C.train_cnn(cfg)

    _, _, split, _ = build_split(cfg)
    assert len(seen["epochs"]) == cfg.cnn.epochs
    held_out = {p for p, _ in split.val} | {p for p, _ in split.test}
    for samples in seen["epochs"]:
        assert sorted(samples) == sorted(split.train)
        assert not held_out & {p for p, _ in samples}
    train_paths = {p for p, _ in split.train}
    assert seen["ref"] and all(set(p) == train_paths for p in seen["ref"])
