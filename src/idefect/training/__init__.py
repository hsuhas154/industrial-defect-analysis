"""Training entry points for the classical-ML and CNN branches.

Submodules are imported lazily via :func:`__getattr__` so that importing the
classical branch (or the synthetic-data generator) does not require PyTorch. Use
``from idefect.training import train_classical_ml`` / ``train_cnn`` as usual.
"""

from __future__ import annotations

__all__ = ["train_classical_ml", "train_cnn"]


def __getattr__(name: str):
    if name == "train_classical_ml":
        from idefect.training.train_classical_ml import train_classical_ml

        return train_classical_ml
    if name == "train_cnn":
        from idefect.training.train_cnn import train_cnn

        return train_cnn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
