#!/usr/bin/env python3
"""Generate every classical-branch figure from saved results.

Usage:
    python scripts/classical_report.py --config configs/neu_baseline.yaml
"""

import _bootstrap  # noqa: F401

from idefect.scripts_entry import classical_report_main

if __name__ == "__main__":
    classical_report_main()
