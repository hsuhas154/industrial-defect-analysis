"""Typed configuration loaded from YAML.

A single YAML file (see ``configs/neu_baseline.yaml``) drives every experiment.
This module parses it into lightweight dataclasses with sensible defaults so the
rest of the code base can rely on attribute access and IDE completion.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class DataConfig:
    root: str = "data/raw/NEU-CLS"
    image_size: int = 128
    grayscale: bool = True
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15


@dataclass
class PreprocessingConfig:
    normalize: bool = True
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    denoise: str = "none"  # none | gaussian | median
    denoise_ksize: int = 3


@dataclass
class HOGConfig:
    orientations: int = 9
    pixels_per_cell: List[int] = field(default_factory=lambda: [16, 16])
    cells_per_block: List[int] = field(default_factory=lambda: [2, 2])
    block_norm: str = "L2-Hys"
    transform_sqrt: bool = True


@dataclass
class DecisionTreeConfig:
    max_depth: int = 20
    min_samples_leaf: int = 2
    criterion: str = "gini"


@dataclass
class SVMConfig:
    kernel: str = "rbf"
    C: float = 10.0
    gamma: str = "scale"
    standardize: bool = True


@dataclass
class ClassicalMLConfig:
    decision_tree: DecisionTreeConfig = field(default_factory=DecisionTreeConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)


@dataclass
class CNNConfig:
    in_channels: int = 1
    base_channels: int = 32
    dropout: float = 0.3
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    augment: bool = True
    device: str = "auto"  # auto | cpu | cuda


@dataclass
class OutputConfig:
    results_dir: str = "experiments/results"
    figures_dir: str = "experiments/figures"
    logs_dir: str = "experiments/logs"


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    hog: HOGConfig = field(default_factory=HOGConfig)
    classical_ml: ClassicalMLConfig = field(default_factory=ClassicalMLConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _from_dict(cls, data: Dict[str, Any]):
    """Recursively build a (possibly nested) dataclass from a plain dict.

    Unknown keys are ignored so the config file can carry extra annotations
    without breaking loading.
    """
    if not is_dataclass(cls):
        return data
    # Resolve string annotations (this module uses ``from __future__ import
    # annotations``, so ``field.type`` would otherwise be a str) to real types.
    hints = typing.get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    data = data or {}
    for f in fields(cls):
        name = f.name
        if name not in data:
            continue
        ftype = hints.get(name, f.type)
        value = data[name]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _from_dict(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file into a :class:`Config`."""
    path = Path(path)
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _from_dict(Config, raw)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    total = cfg.data.train_frac + cfg.data.val_frac + cfg.data.test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"data split fractions must sum to 1.0, got {total:.4f} "
            f"(train={cfg.data.train_frac}, val={cfg.data.val_frac}, "
            f"test={cfg.data.test_frac})"
        )
    if cfg.data.image_size <= 0:
        raise ValueError("data.image_size must be positive")
    if cfg.preprocessing.denoise not in {"none", "gaussian", "median"}:
        raise ValueError(
            "preprocessing.denoise must be one of none|gaussian|median"
        )
