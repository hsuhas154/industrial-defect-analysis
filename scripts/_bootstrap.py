"""Make ``src/`` importable when running scripts without installing the package.

Import this at the top of any script in ``scripts/`` so that
``python scripts/foo.py`` works from the repository root even if the user has
not run ``pip install -e .``.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
