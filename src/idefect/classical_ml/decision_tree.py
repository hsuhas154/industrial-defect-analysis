"""Decision Tree baseline.

A single decision tree is an interpretable, non-linear baseline. It is a useful
conceptual contrast to the margin-based SVM and to the ensemble/boosting methods
that arrive in later phases, and it is a clear illustration of why an unpruned
tree tends to overfit.
"""

from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier

from idefect.config import DecisionTreeConfig


def build_decision_tree(cfg: DecisionTreeConfig, seed: int = 42) -> DecisionTreeClassifier:
    """Construct a configured :class:`DecisionTreeClassifier`."""
    return DecisionTreeClassifier(
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        criterion=cfg.criterion,
        random_state=seed,
    )
