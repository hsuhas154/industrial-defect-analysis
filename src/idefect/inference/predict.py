"""Load a trained model and predict the defect class of a single image.

Handles both artifact kinds produced by the training scripts:

* classical joblib bundles (``*.joblib``): HOG + Decision Tree / SVM;
* CNN checkpoints (``*.pt``): the custom CNN.

The saved artifacts carry their own preprocessing/HOG config so the exact
feature pipeline used at training time is reproduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from idefect.data.preprocessing import load_grayscale, preprocess_image
from idefect.features.hog import extract_hog


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
        self.class_names = bundle["class_names"]
        self.image_size = bundle["image_size"]
        self._hog_cfg = bundle["hog"]
        self._prep_cfg = bundle["preprocessing"]

    def _load_cnn(self) -> None:
        import torch

        from idefect.models.custom_cnn import CustomCNN

        ckpt = torch.load(self.artifact_path, map_location="cpu", weights_only=False)
        self._backend = "cnn"
        self.kind = "custom_cnn"
        self.class_names = ckpt["class_names"]
        self.image_size = ckpt["image_size"]
        self._prep_cfg = ckpt["preprocessing"]
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
        feats = extract_hog(proc, self._hog_cfg).reshape(1, -1)
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
