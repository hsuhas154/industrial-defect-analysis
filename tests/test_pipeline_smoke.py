"""End-to-end smoke tests: train on synthetic data, predict, run the app.

These exercise the real training entry points with tiny settings so a broken
pipeline is caught by ``pytest`` without the NEU download.
"""

import os
from pathlib import Path

import pytest

from idefect.config import Config
from idefect.inference.predict import Predictor

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny_cfg(dummy_dataset, tmp_path_factory) -> Config:
    out = tmp_path_factory.mktemp("runs")
    cfg = Config()
    cfg.data.root = str(dummy_dataset)
    cfg.data.image_size = 32
    cfg.data.train_frac, cfg.data.val_frac, cfg.data.test_frac = 0.5, 0.25, 0.25
    cfg.hog.pixels_per_cell = [8, 8]
    cfg.cnn.epochs = 2
    cfg.cnn.batch_size = 8
    cfg.cnn.num_workers = 0
    cfg.cnn.base_channels = 8
    cfg.cnn.device = "cpu"
    cfg.output.results_dir = str(out / "results")
    cfg.output.figures_dir = str(out / "figures")
    cfg.output.logs_dir = str(out / "logs")
    return cfg


@pytest.fixture(scope="module")
def trained(tiny_cfg):
    from idefect.training.train_classical_ml import train_classical_ml

    metrics = train_classical_ml(tiny_cfg)
    return tiny_cfg, metrics


def test_classical_training_writes_artifacts(trained):
    cfg, metrics = trained
    results = Path(cfg.output.results_dir)
    for kind in ("hog+decision_tree", "hog+svm"):
        assert (results / f"{kind}.joblib").exists()
        assert (results / f"{kind}_metrics.json").exists()
        assert (results / f"{kind}_test_predictions.csv").exists()
        assert 0.0 <= metrics[kind]["macro_f1"] <= 1.0
    assert (Path(cfg.output.logs_dir) / "runs.csv").exists()


def test_predictor_roundtrip_classical(trained, dummy_dataset):
    cfg, _ = trained
    image = next(Path(dummy_dataset).rglob("*.png"))
    pred = Predictor(Path(cfg.output.results_dir) / "hog+svm.joblib").predict(image)
    assert pred.label in pred.probabilities
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-6


def test_cnn_training_and_prediction(tiny_cfg, dummy_dataset):
    pytest.importorskip("torch")
    from idefect.training.train_cnn import train_cnn

    metrics = train_cnn(tiny_cfg)
    assert metrics["epochs_run"] == 2
    ckpt = Path(tiny_cfg.output.results_dir) / "custom_cnn.pt"
    image = next(Path(dummy_dataset).rglob("*.png"))
    pred = Predictor(ckpt).predict(image)
    assert pred.model_kind == "custom_cnn"
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-5


def test_streamlit_app_runs_with_upload(trained, dummy_dataset, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    cfg, _ = trained
    monkeypatch.setenv("IDEFECT_RESULTS_DIR", cfg.output.results_dir)
    at = AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    image = next(Path(dummy_dataset).rglob("*.png"))
    at.file_uploader[0].upload(image.name, image.read_bytes(), "image/png")
    at.sidebar.checkbox[0].check()
    at.run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert "Predicted class" in labels
    assert len(at.dataframe) == 1  # side-by-side comparison of all models
