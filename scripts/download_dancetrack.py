#!/usr/bin/env python
"""DanceTrack (uniform-appearance, diverse-motion benchmark): https://github.com/DanceTrack/DanceTrack

The data is distributed by the authors through links on their GitHub page / project page (cloud-drive
hosting that requires accepting the terms), so it cannot be fetched blindly. Manual steps:

  1. Open https://github.com/DanceTrack/DanceTrack and follow the dataset download links
     (you need the *train* and *val* parts; the test GT is private).
  2. Run:  python scripts/download_dancetrack.py --zips train1.zip train2.zip val.zip
     (any zip names work; folders named train/val are detected) -> data/dancetrack/{train,val}/<seq>/

DanceTrack ships NO detections. Use `--det-source oracle` (noisy GT boxes, "D0") or supply your own
MOT-format det.txt per sequence and use `--det-source file`.
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from rcamot.datasets.download import normalise_dancetrack

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--zips", nargs="*", help="manually downloaded DanceTrack zip files")
ap.add_argument("--root", default="data")
a = ap.parse_args()
if not a.zips:
    print(__doc__)
    sys.exit(0)
try:
    print(f"OK -> {normalise_dancetrack(a.zips, a.root)}\nNext: python scripts/verify_dataset.py --dataset dancetrack")
except RuntimeError as e:
    print(f"ERROR: {e}")
    sys.exit(2)
