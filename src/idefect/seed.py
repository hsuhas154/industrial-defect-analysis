"""Global seeding utilities for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (if available) PyTorch.

    Parameters
    ----------
    seed:
        The random seed.
    deterministic_torch:
        If True and torch is installed, request deterministic cuDNN behaviour.
        This makes CNN runs reproducible at a small performance cost.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is optional for the classical-only path
        pass
