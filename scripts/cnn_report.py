#!/usr/bin/env python3
"""Grad-CAM galleries for every trained CNN and a cross-model comparison.

Usage:
    python scripts/cnn_report.py --config configs/neu_baseline.yaml
"""

import _bootstrap  # noqa: F401

from idefect.scripts_entry import cnn_report_main

if __name__ == "__main__":
    cnn_report_main()
