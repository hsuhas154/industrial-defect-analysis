"""Console-script entry points declared in pyproject.toml.

These thin wrappers let the tool be driven either via ``python scripts/*.py`` or
via the installed console commands (``idefect-train-cnn`` etc.).
"""

from __future__ import annotations

import argparse

from idefect.config import load_config

# NOTE: torch-dependent imports (the CNN trainer, the Predictor's CNN backend)
# are done lazily inside the functions below so the classical branch and the
# synthetic-data generator work even if PyTorch is not installed.


def _add_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        default="configs/neu_baseline.yaml",
        help="Path to the YAML config (default: configs/neu_baseline.yaml)",
    )


def gen_dummy_main() -> None:
    from idefect.data.synthetic import generate_dummy_dataset

    p = argparse.ArgumentParser(description="Generate a synthetic NEU-style dataset.")
    p.add_argument("--out", default="data/raw/NEU-CLS", help="Output directory")
    p.add_argument("--per-class", type=int, default=40, help="Images per class")
    p.add_argument("--size", type=int, default=200, help="Image size (px)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = generate_dummy_dataset(args.out, args.per_class, args.size, args.seed)
    print(f"Wrote synthetic dataset to {out}")


def train_classical_main() -> None:
    from idefect.training.train_classical_ml import train_classical_ml

    p = argparse.ArgumentParser(description="Train HOG + Decision Tree / SVM baselines.")
    _add_config_arg(p)
    args = p.parse_args()
    cfg = load_config(args.config)
    train_classical_ml(cfg)


def train_cnn_main() -> None:
    from idefect.training.train_cnn import train_cnn

    p = argparse.ArgumentParser(description="Train the custom CNN baseline.")
    _add_config_arg(p)
    args = p.parse_args()
    cfg = load_config(args.config)
    train_cnn(cfg)


def predict_main() -> None:
    from idefect.inference.predict import Predictor

    p = argparse.ArgumentParser(description="Predict the defect class of one image.")
    p.add_argument("image", help="Path to the image to classify")
    p.add_argument(
        "--model",
        required=True,
        help="Path to a trained artifact (.joblib for classical, .pt for CNN)",
    )
    args = p.parse_args()
    predictor = Predictor(args.model)
    result = predictor.predict(args.image)
    print(f"model:      {result.model_kind}")
    print(f"prediction: {result.label}")
    if result.confidence is not None:
        print(f"confidence: {result.confidence:.4f}")
    if result.probabilities:
        print("probabilities:")
        for cls, prob in sorted(result.probabilities.items(), key=lambda kv: -kv[1]):
            print(f"  {cls:<18} {prob:.4f}")
