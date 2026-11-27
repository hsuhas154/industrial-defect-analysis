#!/usr/bin/env python3
"""Verify, extract and fingerprint the KolektorSDD and DAGM 2007 downloads.

Usage (first run creates the manifests; later runs only verify):
    python scripts/prepare_phase4_data.py --write-manifests
    python scripts/prepare_phase4_data.py
"""

import _bootstrap  # noqa: F401

from idefect.scripts_entry import prepare_phase4_main

if __name__ == "__main__":
    prepare_phase4_main()
