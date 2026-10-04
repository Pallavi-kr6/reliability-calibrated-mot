#!/usr/bin/env python
"""Measure identity signal in mined MOT candidate pairs (without changing tracker behavior)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from rcamot.inference.pipeline import ensure_model
from rcamot.training.train_rca import mined_pairs
from rcamot.utils import load_config


def auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if not n1 or not n0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mot17")
    ap.add_argument("--embedder", default="colorhist")
    ap.add_argument("--config", default="configs/rca.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config, dataset=args.dataset, overrides={"data": {"embedder": args.embedder}})
    pairs = mined_pairs(cfg, cfg["data"]["eval_split"])
    model = ensure_model(cfg)
    y, frames = pairs["label"].astype(bool), np.exp(pairs["tsu"])
    feats = {k: pairs[k] for k in ("iou", "ctr", "tsu", "d", "occ", "margin", "score")}
    logits = model.logit(feats)
    rows = []
    for subset, mask in (("all", np.ones(len(y), bool)), ("tsu==0", frames == 1), ("tsu>0", frames > 1)):
        sel = mask
        rows.append({"dataset": args.dataset, "embedder": args.embedder, "subset": subset, "n_pairs": int(sel.sum()),
                     "positive_rate": float(y[sel].mean()) if sel.any() else np.nan,
                     "auc_neg_d": auc(-pairs["d"][sel], y[sel]), "auc_iou": auc(pairs["iou"][sel], y[sel]),
                     "auc_neg_ctr": auc(-pairs["ctr"][sel], y[sel]), "auc_rca_logit": auc(logits[sel], y[sel]),
                     "median_d_positive": float(np.median(pairs["d"][sel & y])) if (sel & y).any() else np.nan,
                     "median_d_negative": float(np.median(pairs["d"][sel & ~y])) if (sel & ~y).any() else np.nan})
    out = Path(cfg["results_dir"]) / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"appearance_{args.dataset}_{args.embedder}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
