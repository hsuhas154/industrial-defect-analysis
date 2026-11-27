#!/usr/bin/env python3
"""DAGM shared-content check, many classes at once, with memory guards, a hang watchdog and live status.

Usage (from the repository folder):
    python scripts/phase4_dagm_check.py            # every class not cached yet, plus verify Classes 1 and 2
    python scripts/phase4_dagm_check.py --status   # from a second terminal: the latest status block

Then build the split from the cached results:
    python scripts/phase4_dataset_report.py --only dagm --no-figures
"""

import sys

import _bootstrap  # noqa: F401

from idefect.data.phase4_dagm_parallel import main

if __name__ == "__main__":
    sys.exit(main())
