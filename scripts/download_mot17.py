#!/usr/bin/env python
"""Download MOT17 (official MOTChallenge data, public, research use) into data/MOT17 and keep only the train split
(the test split has no public ground truth).

  python scripts/download_mot17.py                 # download (several GB, resumable) + extract
  python scripts/download_mot17.py --zip MOT17.zip  # use a zip you downloaded manually from motchallenge.net
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from rcamot.datasets.download import fetch_mot

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", default="data")
ap.add_argument("--zip", help="path to an already downloaded MOT17.zip")
ap.add_argument("--include-test", action="store_true", help="also extract the test split (no GT; not used here)")
ap.add_argument("--keep-zip", action="store_true")
a = ap.parse_args()
try:
    out = fetch_mot("MOT17", a.root, a.include_test, a.keep_zip, a.zip)
except RuntimeError as e:
    print(f"ERROR: {e}")
    sys.exit(2)
print(f"OK -> {out}\nNext: python scripts/verify_dataset.py --dataset mot17")
