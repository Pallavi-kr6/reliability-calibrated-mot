"""Cost construction (B0..B3 baselines and RCA) and Hungarian assignment with an accept gate."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .features import PairFeatures

BIG = 1e5


def assign(cost: np.ndarray, valid: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Hungarian on `cost`; pairs with valid==False are forbidden. Returns (matches, unmatched rows, unmatched cols)."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    c = np.where(valid, cost, BIG)
    rows, cols = linear_sum_assignment(c)
    matches = [(int(r), int(k)) for r, k in zip(rows, cols) if valid[r, k]]
    mr = {r for r, _ in matches}
    mc = {k for _, k in matches}
    return matches, [i for i in range(n) if i not in mr], [j for j in range(m) if j not in mc]


def iou_cost(iou: np.ndarray, min_iou: float) -> Tuple[np.ndarray, np.ndarray]:
    return 1.0 - iou, iou >= min_iou


def fused_cost(f: PairFeatures, method: str, cfg: Dict, high_thresh: float) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Baselines B0-B3.

    b0: IoU only.
    b1: C = (1-lam)(1-IoU) + lam*d with a fixed lam.
    b2: b1 with lam scaled by detection confidence (Deep OC-SORT-style heuristic, re-implemented).
    b3: b1 with the appearance term switched off for heavily overlapped pairs (hard IoA gating, FC-Track-style).
    """
    iou, d = f.iou, f.d
    info: Dict = {}
    if method == "b0":
        cost, valid = 1.0 - iou, iou >= cfg["b0_min_iou"]
        valid &= f.mask
        return cost, valid, info
    lam = np.full(iou.shape, float(cfg["fixed_lambda"]))
    if method == "b2":
        lam = lam * np.clip((f.score - high_thresh) / max(1.0 - high_thresh, 1e-6), 0.0, 1.0)
    elif method == "b3":
        lam = np.where(f.occ > cfg["ioa_gate"], 0.0, lam)
    cost = (1.0 - lam) * (1.0 - iou) + lam * d
    valid = f.mask & (cost <= cfg["fused_cost_thresh"]) & ((d <= cfg["app_gate"]) | (lam == 0.0))
    info["lam"] = lam
    return cost, valid, info


def rca_cost(f: PairFeatures, model, cfg: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """RCA: cost = -log P_ij (calibrated posterior); pairs below the acceptance threshold are forbidden."""
    feats = f.as_dict()
    post = model.posterior(feats, calibrated=cfg.get("calibrate", True))
    cost = -np.log(np.clip(post, 1e-9, 1.0))
    valid = f.mask & (post >= cfg["accept_thresh"])
    return cost, valid, {"post": post, "lam": model.lam(feats)}
