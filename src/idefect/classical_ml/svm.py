"""Support Vector Machine baseline.

An RBF-kernel SVM performs margin-based classification on the HOG features. SVMs
are sensitive to feature scaling, so a ``StandardScaler`` is prepended when
``cfg.standardize`` is set. ``probability=True`` is enabled so the inference UI
can display class confidences.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from idefect.config import SVMConfig


def build_svm(cfg: SVMConfig, seed: int = 42) -> Pipeline:
    """Construct an SVM pipeline (optional scaler + SVC)."""
    svc = SVC(
        kernel=cfg.kernel,
        C=cfg.C,
        gamma=cfg.gamma,
        probability=True,
        random_state=seed,
    )
    steps = []
    if cfg.standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("svc", svc))
    return Pipeline(steps)
