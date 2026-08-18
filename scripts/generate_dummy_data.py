#!/usr/bin/env python3
"""Generate a small synthetic NEU-style dataset for smoke-testing the pipeline.

Usage:
    python scripts/generate_dummy_data.py --out data/raw/NEU-CLS --per-class 40
"""

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

from idefect.scripts_entry import gen_dummy_main

if __name__ == "__main__":
    gen_dummy_main()
