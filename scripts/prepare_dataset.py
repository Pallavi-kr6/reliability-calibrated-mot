#!/usr/bin/env python
"""Create expected directories and (for `synthetic`) generate the demo data; for real datasets print what is missing.

  python scripts/prepare_dataset.py --dataset synthetic [--quick]
  python scripts/prepare_dataset.py --dataset mot17
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from rcamot import experiments as ex
from rcamot.datasets.registry import DatasetMissingError
from rcamot.utils import ensure_dir, load_config

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dataset", required=True, choices=["mot17", "mot20", "dancetrack", "synthetic"])
ap.add_argument("--quick", action="store_true")
ap.add_argument("--regen", action="store_true")
a = ap.parse_args()
for d in ("data", "data/cache", "models/params", "results", "weights"):
    ensure_dir(d)
cfg = load_config("configs/rca.yaml", dataset=a.dataset)
try:
    ex.cmd_prepare(cfg, a.quick, a.regen)
except DatasetMissingError as e:
    print(f"ERROR: {e}")
    sys.exit(2)
