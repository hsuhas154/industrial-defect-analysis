#!/usr/bin/env python3
"""Quantify near-duplicate leakage: random split versus group-aware split.

Usage:
    python scripts/leakage_study.py --config configs/neu_baseline.yaml
"""

import _bootstrap  # noqa: F401

from idefect.scripts_entry import leakage_study_main

if __name__ == "__main__":
    leakage_study_main()
