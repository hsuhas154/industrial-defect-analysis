#!/usr/bin/env python3
"""Dataset statistics, sanity checks and exploratory figures.

Usage:
    python scripts/dataset_report.py --config configs/neu_baseline.yaml
"""

import _bootstrap  # noqa: F401

from idefect.scripts_entry import dataset_report_main

if __name__ == "__main__":
    dataset_report_main()
