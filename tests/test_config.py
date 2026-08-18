"""Tests for config loading and validation."""

from pathlib import Path

import pytest
import yaml

from idefect.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_baseline_config():
    cfg = load_config(REPO_ROOT / "configs" / "neu_baseline.yaml")
    assert isinstance(cfg, Config)
    assert cfg.data.image_size > 0
    assert abs(
        cfg.data.train_frac + cfg.data.val_frac + cfg.data.test_frac - 1.0
    ) < 1e-6
    assert cfg.hog.orientations == 9
    assert cfg.cnn.epochs > 0


def test_bad_fractions_raise(tmp_path):
    bad = {
        "data": {"train_frac": 0.5, "val_frac": 0.5, "test_frac": 0.5},
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError):
        load_config(p)
