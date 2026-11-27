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
class DuplicateConfig:
    """Near-duplicate detection settings (see idefect.data.duplicates)."""

    enabled: bool = True
    # E1: exact overlaps
    segment_len: int = 12
    segment_stride: int = 3
    min_segment_range: int = 10      # skip flat runs (max - min below this)
    max_key_occurrences: int = 40    # skip non-discriminative runs
    min_votes: int = 4
    exact_tolerance: int = 2
    exact_fraction: float = 0.99
    min_exact_overlap_px: int = 1024
    # E2: re-processed copies
    thumb_size: int = 64
    top_k: int = 10
    min_overlap: float = 0.25
    grain_same: float = 0.50
    grain_review: float = 0.40
    grain_min_tiles: int = 12
    grain_tile: int = 16
    grain_mask_grow: int = 12
    # misc
    device: str = "auto"
    review_file: str = "configs/duplicate_review.yaml"
    cache_dir: str = "experiments/cache"


@dataclass
class DataConfig:
    root: str = "data/raw/NEU-CLS"
    manifest: str = "data/neu_cls_manifest.sha256"  # SHA-256 of every image; "" to skip
    image_size: int = 128
    grayscale: bool = True
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    split_seed: int = 42  # fixed so every model and every training seed shares one split
    duplicates: DuplicateConfig = field(default_factory=DuplicateConfig)


@dataclass
class ContentScreenConfig:
    """Shared-content checks for the Phase 4 datasets (idefect.data.phase4_leakage).

    E1 (pixel identity, any flip, rotation or shift) runs with the NEU
    settings in data.duplicates, except for the reference-run stride, raised so
    that memory stays bounded on large images. Re-processed copies are screened by masked
    NCC of band-passed images: coarse over all pairs, fine for the candidates;
    the fine score decides. Planted copies prove per dataset (per DAGM class)
    that a copy would be caught; the run stops otherwise.
    """
    exact_stride: int = 6          # E1 reference-run stride (NEU uses 3 on 200 px images)
    coarse_factor: float = 8.0     # coarse screen: images reduced by this factor
    fine_factor: float = 2.0       # fine verification of the candidates
    top_k: int = 10                # candidates: each image's best coarse partners ...
    candidate_score: float = 0.9   # ... and every pair at or above this coarse score
    same: float = 0.5              # fine score: same content (grouped)
    review: float = 0.3            # fine score in [review, same): grouped and shown to a person
    min_overlap: float = 0.5       # shifts count only with this much overlap
    planted_sources: int = 40      # images used to plant re-processed copies


@dataclass
class KolektorConfig:
    """KolektorSDD (fine annotations): pixel masks for defect localization."""
    root: str = "data/raw/KolektorSDD"       # holds kos01 ... kos50 and splits/
    manifest: str = "data/kolektorsdd_manifest.sha256"
    splits_file: str = "data/raw/KolektorSDD/splits/split.pyb"  # official 3-fold split (by part)
    duplicate_policy: str = "drop_copies"   # drop_copies | keep_together (byte-identical parts)
    val_parts_frac: float = 0.2             # validation parts carved from each fold's training parts
    split_seed: int = 42
    screen: ContentScreenConfig = field(default_factory=lambda: ContentScreenConfig(
        exact_stride=6, coarse_factor=5.0))


@dataclass
class DAGMConfig:
    """DAGM 2007: normal versus defective textures with weak ellipse labels."""
    root: str = "data/raw/DAGM"             # holds Class1 ... Class10
    manifest: str = "data/dagm_manifest.sha256"
    classes: List[int] = field(default_factory=lambda: list(range(1, 11)))
    val_frac: float = 0.2                   # validation images carved from the official Train side
    split_seed: int = 42
    screen: ContentScreenConfig = field(default_factory=lambda: ContentScreenConfig(
        exact_stride=10, coarse_factor=8.0))
    review_file: str = "configs/phase4_review.yaml"   # human decisions for review-band pairs


@dataclass
class PreprocessingConfig:
    normalize: bool = True
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    denoise: str = "none"  # none | gaussian | median
    denoise_ksize: int = 3
    enhance: str = "none"  # none | sobel | tophat (edge or morphology enhanced input)
    enhance_ksize: int = 9  # structuring element size for tophat


@dataclass
class HOGConfig:
    orientations: int = 9
    pixels_per_cell: List[int] = field(default_factory=lambda: [16, 16])
    cells_per_block: List[int] = field(default_factory=lambda: [2, 2])
    block_norm: str = "L2-Hys"
    transform_sqrt: bool = True


@dataclass
class LBPConfig:
    points_radii: List[List[float]] = field(default_factory=lambda: [[8, 1], [16, 2], [24, 3]])
    method: str = "uniform"  # rotation-invariant uniform patterns (P + 2 codes)
    grid: int = 1            # 1 = global histogram; 2 = 2x2 cells, ...


@dataclass
class GLCMConfig:
    levels: int = 32
    distances: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    properties: List[str] = field(default_factory=lambda: [
        "contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"])


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
    feature_sets: List[str] = field(default_factory=lambda: [
        "hog", "lbp", "glcm", "lbp+glcm", "hog+lbp+glcm"])
    models: List[str] = field(default_factory=lambda: [
        "decision_tree", "logreg", "knn", "svm", "random_forest", "xgboost"])
    cv_folds: int = 5
    n_jobs: int = 8            # worker processes for the grid searches and feature extraction
    xgb_device: str = "auto"   # auto | cuda | cpu
    # Untuned reference settings (kept for the original HOG baselines).
    decision_tree: DecisionTreeConfig = field(default_factory=DecisionTreeConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)


@dataclass
class CNNConfig:
    arch: str = "custom_cnn"  # custom_cnn | vgg | resnet | densenet | mobilenet (models.registry)
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
    scheduler: str = "cosine"  # cosine | plateau | none
    select_metric: str = "macro_f1"  # macro_f1 | accuracy | loss (checkpoint selection)
    early_stopping_patience: int = 0  # epochs without improvement; 0 disables
    cache_images: bool = True  # keep preprocessed images in memory
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    finetune: str = "full"  # ImageNet backbones only: frozen | partial | full
    pretrained: bool = True  # ImageNet backbones: start from ImageNet weights
    backbone_lr_mult: float = 0.1  # learning-rate factor for unfrozen pretrained layers
    amp: bool = False  # bfloat16 mixed precision on CUDA (off: full precision, as in all reported runs)
    tensorboard: bool = False  # also write per-epoch curves to <logs_dir>/tensorboard


@dataclass
class OutputConfig:
    results_dir: str = "experiments/results"
    figures_dir: str = "experiments/figures"
    logs_dir: str = "experiments/logs"


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    kolektorsdd: KolektorConfig = field(default_factory=KolektorConfig)
    dagm: DAGMConfig = field(default_factory=DAGMConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    hog: HOGConfig = field(default_factory=HOGConfig)
    lbp: LBPConfig = field(default_factory=LBPConfig)
    glcm: GLCMConfig = field(default_factory=GLCMConfig)
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
    if cfg.kolektorsdd.duplicate_policy not in {"drop_copies", "keep_together"}:
        raise ValueError("kolektorsdd.duplicate_policy must be drop_copies or keep_together")
    for name, frac in (("kolektorsdd.val_parts_frac", cfg.kolektorsdd.val_parts_frac),
                       ("dagm.val_frac", cfg.dagm.val_frac)):
        if not 0.0 < frac < 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    for name, sc in (("kolektorsdd.screen", cfg.kolektorsdd.screen), ("dagm.screen", cfg.dagm.screen)):
        if not 0.0 < sc.review <= sc.same <= 1.0:
            raise ValueError(f"{name}: need 0 < review <= same <= 1")
        if sc.exact_stride < 1 or sc.top_k < 1 or sc.coarse_factor < sc.fine_factor:
            raise ValueError(f"{name}: exact_stride and top_k must be positive, coarse_factor >= fine_factor")
    if not set(cfg.dagm.classes) <= set(range(1, 11)) or not cfg.dagm.classes:
        raise ValueError("dagm.classes must be a non-empty subset of 1..10")
    if cfg.preprocessing.enhance not in {"none", "sobel", "tophat"}:
        raise ValueError("preprocessing.enhance must be one of none|sobel|tophat")
    if cfg.cnn.finetune not in {"frozen", "partial", "full"}:
        raise ValueError("cnn.finetune must be one of frozen|partial|full")
    if cfg.cnn.scheduler not in {"cosine", "plateau", "none"}:
        raise ValueError("cnn.scheduler must be one of cosine|plateau|none")
    if cfg.cnn.select_metric not in {"macro_f1", "accuracy", "loss"}:
        raise ValueError("cnn.select_metric must be one of macro_f1|accuracy|loss")
    if cfg.cnn.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("cnn.device must be one of auto|cpu|cuda")
