"""Classical classifiers and their hyperparameter search spaces.

Every model is an sklearn estimator (a Pipeline when it needs feature
scaling), so scaling is fitted inside each cross-validation fold on that
fold's training part only. The search spaces are deliberately modest but
cover the parameter that matters most for each model:

* Decision Tree: depth, leaf size and cost-complexity pruning (overfitting).
* Logistic Regression: inverse regularization strength C (linear baseline).
* k-NN: number of neighbours and distance weighting (instance-based baseline).
* SVM: RBF kernel width gamma and margin softness C, plus a linear kernel.
* Random Forest: features per split, depth and leaf size (variance reduction
  by bagging decorrelated trees).
* XGBoost: learning rate, depth, column subsampling (bias reduction by
  boosting), trained on the GPU.

``gamma`` for the RBF-SVM is expressed relative to sklearn's "scale" value
(1 / n_features on standardized data), so the same grid fits any feature set.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

MODEL_NAMES = ["decision_tree", "logreg", "knn", "svm", "random_forest", "xgboost"]

DISPLAY = {
    "decision_tree": "Decision Tree",
    "logreg": "Logistic Regression",
    "knn": "k-NN",
    "svm": "SVM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def xgboost_available_on_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def build_model(name: str, n_features: int, seed: int, xgb_device: str = "auto"):
    """Return ``(estimator, param_grid, parallel_ok)`` for a model name.

    ``parallel_ok`` is False for estimators that should not be fitted by
    several processes at once (XGBoost on a single GPU).
    """
    if name == "decision_tree":
        est = DecisionTreeClassifier(random_state=seed)
        grid = {
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_leaf": [1, 2, 5, 10],
            "criterion": ["gini", "entropy"],
            "ccp_alpha": [0.0, 0.001, 0.005],
        }
        return est, grid, True
    if name == "logreg":
        est = Pipeline([("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=5000, random_state=seed))])
        grid = {"clf__C": list(np.logspace(-3, 2, 6))}
        return est, grid, True
    if name == "knn":
        est = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])
        grid = {"clf__n_neighbors": [1, 3, 5, 7, 11, 15],
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "manhattan"]}
        return est, grid, True
    if name == "svm":
        est = Pipeline([("scaler", StandardScaler()),
                        ("clf", SVC(kernel="rbf", random_state=seed))])
        base = 1.0 / max(1, n_features)
        grid = [
            {"clf__kernel": ["rbf"], "clf__C": [0.3, 1, 3, 10, 30, 100],
             "clf__gamma": [base * f for f in (0.1, 0.3, 1.0, 3.0)]},
            {"clf__kernel": ["linear"], "clf__C": [0.001, 0.01, 0.1, 1]},
        ]
        return est, grid, True
    if name == "random_forest":
        est = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=1)
        grid = {"max_features": ["sqrt", 0.1, 0.3],
                "max_depth": [None, 20],
                "min_samples_leaf": [1, 3]}
        return est, grid, True
    if name == "xgboost":
        from xgboost import XGBClassifier

        device = "cuda" if (xgb_device == "auto" and xgboost_available_on_gpu()) else (
            "cpu" if xgb_device == "auto" else xgb_device)
        est = XGBClassifier(n_estimators=400, tree_method="hist", device=device,
                            subsample=0.8, eval_metric="mlogloss", random_state=seed,
                            n_jobs=1 if device == "cuda" else -1, verbosity=0)
        grid = {"learning_rate": [0.05, 0.15],
                "max_depth": [4, 6],
                "colsample_bytree": [0.3, 0.7]}
        return est, grid, device != "cuda"
    raise ValueError(f"unknown model '{name}'; choose from {MODEL_NAMES}")


def with_probabilities(estimator):
    """Enable class probabilities where they are optional (SVC), after the
    search, so the search itself stays fast."""
    if isinstance(estimator, Pipeline) and isinstance(estimator.steps[-1][1], SVC):
        estimator.set_params(clf__probability=True)
    return estimator


def grid_size(grid) -> int:
    grids: List[Dict] = grid if isinstance(grid, list) else [grid]
    total = 0
    for g in grids:
        n = 1
        for v in g.values():
            n *= len(v)
        total += n
    return total


def display_name(feature_set: str, model: str) -> Tuple[str, str]:
    return feature_set.upper().replace("+", " + "), DISPLAY.get(model, model)
