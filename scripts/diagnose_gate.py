#!/usr/bin/env python
"""Evaluate current RCA acceptance behavior by time since last association."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from rcamot.inference.pipeline import ensure_model
from rcamot.models.rca import GAP_BINS, gap_bin_mask
from rcamot.training.train_rca import mined_pairs
from rcamot.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mot17")
    ap.add_argument("--embedder", default="colorhist")
    ap.add_argument("--config", default="configs/rca.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config, dataset=args.dataset, overrides={"data": {"embedder": args.embedder}})
    model = ensure_model(cfg)
    pairs = mined_pairs(cfg, cfg["data"]["eval_split"])
    y = pairs["label"].astype(bool)
    frames = np.exp(pairs["tsu"])
    feats = {k: pairs[k] for k in ("iou", "ctr", "tsu", "d", "occ", "margin", "score")}
    p = model.posterior(feats, calibrated=cfg["rca"].get("calibrate", True))
    thresholds = model.threshold(feats, cfg["rca"]["accept_thresh"]) if cfg["rca"].get("gate_mode") == "far_per_gap" else np.full(len(y), cfg["rca"]["accept_thresh"])
    rows = []
    for key, _, _ in GAP_BINS:
        mask = gap_bin_mask(frames, key)
        pos, neg = mask & y, mask & ~y
        rows.append({"dataset": args.dataset, "gap_bin": key, "n_pairs": int(mask.sum()), "n_positive": int(pos.sum()),
                     "n_negative": int(neg.sum()), "tau": float(np.median(thresholds[mask])) if mask.any() else np.nan,
                     "positive_rejection_rate": float((p[pos] < thresholds[pos]).mean()) if pos.any() else np.nan,
                     "negative_acceptance_rate": float((p[neg] >= thresholds[neg]).mean()) if neg.any() else np.nan,
                     "median_positive_posterior": float(np.median(p[pos])) if pos.any() else np.nan})
    out = Path(cfg["results_dir"]) / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"gate_{args.dataset}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
