#!/usr/bin/env python
"""Verify a dataset layout: sequences exist, seqinfo/gt/det parse, image count matches, frame ranges are sane.
Exit code 0 = OK, 2 = problems (with instructions)."""
import argparse
import sys

import _bootstrap  # noqa: F401
from rcamot.datasets.mot_io import load_det, load_gt
from rcamot.datasets.registry import DatasetMissingError, HELP, list_sequences

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dataset", required=True, choices=["mot17", "mot20", "dancetrack", "synthetic"])
ap.add_argument("--root", default="data")
ap.add_argument("--mot17-detector", default="FRCNN", choices=["FRCNN", "DPM", "SDP"])
a = ap.parse_args()
bad = 0
try:
    for split in ("train", "val"):
        for s in list_sequences(a.dataset, split, a.root, a.mot17_detector):
            probs = []
            if not s.gt_path.exists():
                probs.append("gt.txt missing")
            else:
                g = load_gt(s.gt_path)
                if len(g) == 0 or g[:, 0].max() < s.frame_end:
                    probs.append(f"gt covers frames up to {g[:, 0].max() if len(g) else 0} < {s.frame_end}")
            n_img = len(list(s.img_dir.glob(f"*{s.ext}"))) if s.img_dir.exists() else 0
            if n_img == 0:
                probs.append(f"no images in {s.img_dir}")
            elif n_img < s.frame_end:
                probs.append(f"only {n_img} images, need >= {s.frame_end}")
            if s.det_path.exists():
                load_det(s.det_path)
            print(f"[{'OK ' if not probs else 'BAD'}] {a.dataset}/{split}/{s.name}: frames {s.frame_start}-{s.frame_end}, images={n_img}, "
                  f"det={'yes' if s.det_path.exists() else 'no'}" + (f"  <- {'; '.join(probs)}" if probs else ""))
            bad += bool(probs)
except DatasetMissingError as e:
    print(f"ERROR: {e}")
    sys.exit(2)
sys.exit(2 if bad else 0)
