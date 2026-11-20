"""Load a trained model and predict the defect class of a single image.

Handles both artifact kinds produced by the training scripts:

* classical joblib bundles (``*.joblib``): a feature set (HOG, LBP, GLCM or
  a combination) with a classifier (Decision Tree, logistic regression,
  k-NN, SVM, Random Forest or XGBoost);
* CNN checkpoints (``*.pt``): the custom CNN.

The saved artifacts carry their own preprocessing and feature configs so the
exact feature pipeline used at training time is reproduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from idefect.data.preprocessing import load_grayscale, preprocess_image
from idefect.features.registry import extract_one


def n_neighbors_of(model) -> Optional[int]:
    """Number of neighbours of a k-NN model (bare or the ``clf`` step of a
    pipeline); None for any other model."""
    steps = getattr(model, "named_steps", None)
    clf = steps["clf"] if steps is not None and "clf" in steps else model
    k = getattr(clf, "n_neighbors", None)
    return int(k) if k is not None else None


@dataclass
class PredictionResult:
    label: str
    index: int
    confidence: Optional[float]  # None if the model cannot produce probabilities
    probabilities: Optional[Dict[str, float]]
    model_kind: str


class Predictor:
    """Wraps a single trained artifact and exposes ``predict``."""

    def __init__(self, artifact_path: str | Path):
        self.artifact_path = Path(artifact_path)
        self.kind: str = ""
        self.class_names: List[str] = []
        self.image_size: int = 128
        self._backend: str = ""  # "classical" | "cnn"
        self.reference: dict = {}
        self.n_neighbors: Optional[int] = None  # set for k-NN models
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        suffix = self.artifact_path.suffix.lower()
        if suffix == ".joblib":
            self._load_classical()
        elif suffix in {".pt", ".pth"}:
            self._load_cnn()
        else:
            raise ValueError(
                f"Unsupported artifact type '{suffix}'. Expected .joblib or .pt"
            )

    def _load_classical(self) -> None:
        import joblib

        bundle = joblib.load(self.artifact_path)
        self._backend = "classical"
        self.kind = bundle["kind"]
        self.model = bundle["model"]
        self.n_neighbors = n_neighbors_of(self.model)
        self.class_names = bundle["class_names"]
        self.image_size = bundle["image_size"]
        self._hog_cfg = bundle["hog"]
        self._prep_cfg = bundle["preprocessing"]
        self.reference = bundle.get("reference", {})
        # Bundles from the feature-registry trainer name their feature set;
        # older bundles are plain HOG.
        self.feature_set = bundle.get("feature_set", "hog")
        from idefect.config import Config

        fcfg = Config()
        fcfg.hog = bundle["hog"]
        if "lbp" in bundle:
            fcfg.lbp = bundle["lbp"]
        if "glcm" in bundle:
            fcfg.glcm = bundle["glcm"]
        self._feature_cfg = fcfg

    def _load_cnn(self) -> None:
        import torch

        from idefect.models.custom_cnn import CustomCNN

        ckpt = torch.load(self.artifact_path, map_location="cpu", weights_only=False)
        self._backend = "cnn"
        self.kind = "custom_cnn"
        self.class_names = ckpt["class_names"]
        self.image_size = ckpt["image_size"]
        self._prep_cfg = ckpt["preprocessing"]
        self.reference = ckpt.get("reference", {})
        cnn_cfg = ckpt["cnn_config"]
        model = CustomCNN(
            num_classes=ckpt["num_classes"],
            in_channels=cnn_cfg.get("in_channels", 1),
            base_channels=cnn_cfg.get("base_channels", 32),
            dropout=cnn_cfg.get("dropout", 0.3),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self.model = model
        self._torch = torch

    # --------------------------------------------------------------- predict
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply the exact preprocessing used at training time."""
        return preprocess_image(np.asarray(image), self._prep_cfg, self.image_size)

    def predict(self, image) -> PredictionResult:
        """Predict from an image path or a grayscale/color NumPy array."""
        if isinstance(image, (str, Path)):
            image = load_grayscale(image)
        image = np.asarray(image)
        proc = self.preprocess(image)

        if self._backend == "classical":
            return self._predict_classical(proc)
        return self._predict_cnn(proc)

    def _predict_classical(self, proc: np.ndarray) -> PredictionResult:
        feats = extract_one(proc, self.feature_set, self._feature_cfg).reshape(1, -1)
        idx = int(self.model.predict(feats)[0])
        probs = None
        conf = None
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(feats)[0]
            probs = {self.class_names[i]: float(p[i]) for i in range(len(p))}
            conf = float(p[idx])
        return PredictionResult(
            label=self.class_names[idx],
            index=idx,
            confidence=conf,
            probabilities=probs,
            model_kind=self.kind,
        )

    def _predict_cnn(self, proc: np.ndarray) -> PredictionResult:
        torch = self._torch
        tensor = torch.from_numpy(proc).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
        with torch.no_grad():
            logits = self.model(tensor)
            p = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(p.argmax())
        probs = {self.class_names[i]: float(p[i]) for i in range(len(p))}
        return PredictionResult(
            label=self.class_names[idx],
            index=idx,
            confidence=float(p[idx]),
            probabilities=probs,
            model_kind=self.kind,
        )
