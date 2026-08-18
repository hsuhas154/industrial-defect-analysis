"""Classical machine-learning baselines on top of handcrafted features."""

from idefect.classical_ml.decision_tree import build_decision_tree
from idefect.classical_ml.svm import build_svm

__all__ = ["build_decision_tree", "build_svm"]
