#!/usr/bin/env python3
"""Classical-branch ablations: preprocessing, resolution and HOG parameters.

Usage:
    python scripts/run_ablations.py --config configs/neu_baseline.yaml
"""

import _bootstrap  # noqa: F401

from idefect.scripts_entry import ablations_main

if __name__ == "__main__":
    ablations_main()
