"""Shared pytest fixtures.

Adds ``src/`` to the path and provides a tiny synthetic dataset so the tests run
without the real NEU database and without installing the package.
"""

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from idefect.data.synthetic import generate_dummy_dataset  # noqa: E402


@pytest.fixture(scope="session")
def dummy_dataset(tmp_path_factory) -> Path:
    """A small synthetic NEU-style dataset (6 classes × 8 images)."""
    root = tmp_path_factory.mktemp("neu_dummy")
    generate_dummy_dataset(root, per_class=8, size=64, seed=1)
    return root
