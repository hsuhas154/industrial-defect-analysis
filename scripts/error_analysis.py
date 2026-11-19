#!/usr/bin/env python3
"""Error analysis (misclassified galleries, confused pairs) from saved predictions.

Usage:
    python scripts/error_analysis.py --config configs/neu_baseline.yaml
"""

import _bootstrap  # noqa: F401

from idefect.scripts_entry import error_analysis_main

if __name__ == "__main__":
    error_analysis_main()
