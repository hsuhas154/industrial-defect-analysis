#!/usr/bin/env python3
"""Phase 4 datasets: integrity, shared-content checks and leakage-safe splits.

Usage (run scripts/phase4_dagm_check.py first: it computes the slow DAGM checks in parallel
and caches them; this script then only reads the cache):
    python scripts/phase4_dataset_report.py
    python scripts/phase4_dataset_report.py --only kolektorsdd
"""

import _bootstrap  # noqa: F401

from idefect.scripts_entry import phase4_report_main

if __name__ == "__main__":
    phase4_report_main()
