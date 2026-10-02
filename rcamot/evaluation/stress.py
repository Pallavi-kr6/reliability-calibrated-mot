"""Stress protocols.

S1  natural crowding/occlusion bins: split each sequence into fixed windows, compute the crowding statistic
    kappa (mean #GT box pairs with IoU>0.4 per frame) and the occluded fraction (GT visibility<0.5), bin windows
    into kappa quantiles, evaluate every window as a pseudo-sequence, report per-bin means and paired-bootstrap CIs.
S2  controlled disappearance injection: remove one identity's detections for k frames; measure whether the first
    reappearing detection regains the ORIGINAL id (re-association), takes ANOTHER person's id (wrong-ID = identity
    drift), or a brand-new id (fragmentation / lost).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..algorithm.geometry import iou_matrix
from ..datasets.mot_io import xywh_to_xyxy
from ..inference.cache import DetSet, gt_eval_rows
from ..inference.pipeline import run_tracker
from ..utils import get_logger
from .bootstrap import paired_bootstrap
from .metrics import evaluate

log = get_logger()
S1_METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW")


def _ped(gt: np.ndarray) -> np.ndarray:
    return gt[(gt[:, 6] != 0) & (gt[:, 7] == 1)]


def window_stats(gt: np.ndarray, w0: int, w1: int) -> Dict[str, float]:
    g = _ped(gt)
    g = g[(g[:, 0] >= w0) & (g[:, 0] <= w1)]
    if len(g) == 0:
        return {"kappa": 0.0, "occ_frac": 0.0, "density": 0.0}
    pairs, n = [], 0
    for f in np.unique(g[:, 0]):
        b = xywh_to_xyxy(g[g[:, 0] == f][:, 2:6])
        n += 1
        if len(b) > 1:
            m = iou_matrix(b, b)
            pairs.append(((m > 0.4).sum() - len(b)) / 2)
        else:
            pairs.append(0)
    return {"kappa": float(np.mean(pairs)), "occ_frac": float((g[:, 8] < 0.5).mean()), "density": len(g) / max(n, 1)}


def s1_windows(preds_by_method: Dict[str, Dict[str, np.ndarray]], detsets: List[DetSet], window: int, thr: float = 0.5) -> pd.DataFrame:
    rows = []
    for ds in detsets:
        gt = gt_eval_rows(ds.spec)
        starts = list(range(ds.spec.frame_start, ds.spec.frame_end + 1, window))
        for w0 in starts:
            w1 = min(w0 + window - 1, ds.spec.frame_end)
            if w1 - w0 + 1 < window // 2:
                continue
            st = window_stats(gt, w0, w1)
            if st["density"] == 0:
                continue
            for m, preds in preds_by_method.items():
                r = evaluate(gt, preds[ds.spec.name], (w0, w1), thr)
                rows.append({"seq": ds.spec.name, "w0": w0, "w1": w1, "method": m, **st, **{k: r[k] for k in S1_METRICS}})
    return pd.DataFrame(rows)


def add_bins(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    w = df[["seq", "w0", "kappa"]].drop_duplicates()
    ranks = w["kappa"].rank(method="first")
    w["bin"] = pd.qcut(ranks, q=min(n_bins, len(w)), labels=[f"Q{i + 1}" for i in range(min(n_bins, len(w)))])
    return df.merge(w[["seq", "w0", "bin"]], on=["seq", "w0"])


def s1_tables(df: pd.DataFrame, reference: str, target: str, n_boot: int, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agg = (df.groupby(["bin", "method"], observed=True)
             .agg(n_windows=("w0", "count"), kappa=("kappa", "mean"), occ_frac=("occ_frac", "mean"),
                  HOTA=("HOTA", "mean"), DetA=("DetA", "mean"), AssA=("AssA", "mean"), IDF1=("IDF1", "mean"),
                  MOTA=("MOTA", "mean"), IDSW=("IDSW", "sum")).reset_index())
    diffs = []
    if reference in df["method"].unique() and target in df["method"].unique():
        for b, g in df.groupby("bin", observed=True):
            a = g[g.method == target].set_index(["seq", "w0"])
            r = g[g.method == reference].set_index(["seq", "w0"])
            idx = a.index.intersection(r.index)
            for m in ("HOTA", "AssA", "IDF1", "IDSW"):
                ci = paired_bootstrap((a.loc[idx, m] - r.loc[idx, m]).to_numpy(), n_boot, seed)
                diffs.append({"bin": b, "metric": m, "comparison": f"{target}-{reference}", **ci})
    return agg, pd.DataFrame(diffs)


# ---------------------------------------------------------------- S2: disappearance injection
def _runs(frames: np.ndarray) -> Dict[int, int]:
    return {int(f): i for i, f in enumerate(frames)}


def pick_gaps(ds: DetSet, k: int, n_gaps: int, rng: np.random.Generator, pre: int = 10, post: int = 10) -> List[Tuple[int, int]]:
    """Choose (gt_id, t0) such that the identity has detections on every frame of [t0-pre, t0+k+post]."""
    out = []
    ids = [g for g in np.unique(ds.gt_id) if g >= 0]
    rng.shuffle(ids)
    for g in ids:
        fr = set(ds.frames[ds.gt_id == g].tolist())
        lo, hi = ds.spec.frame_start + pre, ds.spec.frame_end - k - post
        cands = [t for t in range(lo, hi + 1) if all((x in fr) for x in range(t - pre, t + k + post + 1))]
        if cands:
            out.append((int(g), int(rng.choice(cands))))
        if len(out) >= n_gaps:
            break
    return out


def _best_pred(preds_f: np.ndarray, box_xyxy: np.ndarray, thr: float = 0.5):
    if len(preds_f) == 0:
        return None
    iou = iou_matrix(box_xyxy[None], xywh_to_xyxy(preds_f[:, 2:6]))[0]
    j = int(np.argmax(iou))
    return int(preds_f[j, 1]) if iou[j] >= thr else None


def gap_injection(cfg: Dict, ds: DetSet, model, k: int, n_gaps: int, seed: int) -> Dict[str, int]:
    rng = np.random.default_rng(seed * 7919 + k)
    gaps = pick_gaps(ds, k, n_gaps, rng)
    drop = np.zeros(len(ds.frames), bool)
    for g, t0 in gaps:
        drop |= (ds.gt_id == g) & (ds.frames >= t0) & (ds.frames < t0 + k)
    rows, _ = run_tracker(ds.drop(drop), cfg, model)
    gt = _ped(gt_eval_rows(ds.spec))
    pf = {int(f): rows[rows[:, 0] == f] for f in np.unique(rows[:, 0])} if len(rows) else {}
    gf = {int(f): gt[gt[:, 0] == f] for f in np.unique(gt[:, 0])}

    def gt_box(g, f):
        r = gf.get(f)
        if r is None:
            return None
        r = r[r[:, 1] == g]
        return xywh_to_xyxy(r[0, 2:6]) if len(r) else None

    c = {"n_gaps": 0, "invalid": 0, "success": 0, "wrong_id": 0, "new_id": 0, "lost": 0}
    for g, t0 in gaps:
        b0 = gt_box(g, t0 - 1)
        p0 = _best_pred(pf.get(t0 - 1, np.zeros((0, 7))), b0) if b0 is not None else None
        if p0 is None:
            c["invalid"] += 1
            continue
        c["n_gaps"] += 1
        p1 = None
        for f in range(t0 + k, t0 + k + 11):
            b1 = gt_box(g, f)
            if b1 is None:
                continue
            p1 = _best_pred(pf.get(f, np.zeros((0, 7))), b1)
            if p1 is not None:
                break
        if p1 is None:
            c["lost"] += 1
        elif p1 == p0:
            c["success"] += 1
        else:
            other = False
            for f in range(max(ds.spec.frame_start, t0 - 30), t0):
                pr, gr = pf.get(f), gf.get(f)
                if pr is None or gr is None:
                    continue
                sel = pr[pr[:, 1] == p1]
                if len(sel):
                    iou = iou_matrix(xywh_to_xyxy(sel[:, 2:6]), xywh_to_xyxy(gr[:, 2:6]))
                    if any(iou[0, j] >= 0.5 and gr[j, 1] != g for j in range(len(gr))):
                        other = True
                        break
            c["wrong_id" if other else "new_id"] += 1
    return c


def s2_table(cfg_by_method: Dict[str, Dict], models: Dict, detsets: List[DetSet], gap_lengths, n_gaps: int, seeds) -> pd.DataFrame:
    rows = []
    for m, cfg in cfg_by_method.items():
        for k in gap_lengths:
            tot = {"n_gaps": 0, "invalid": 0, "success": 0, "wrong_id": 0, "new_id": 0, "lost": 0}
            for sd in seeds:
                for ds in detsets:
                    c = gap_injection(cfg, ds, models.get(m), k, n_gaps, sd)
                    for kk in tot:
                        tot[kk] += c[kk]
            n = max(tot["n_gaps"], 1)
            rows.append({"method": m, "gap_frames": k, **tot, "reassoc_rate": tot["success"] / n,
                         "wrong_id_rate": tot["wrong_id"] / n, "new_id_rate": tot["new_id"] / n, "lost_rate": tot["lost"] / n})
            log.info(f"[S2] {m} k={k}: n={tot['n_gaps']} success={tot['success']} wrong={tot['wrong_id']} new={tot['new_id']} lost={tot['lost']}")
    return pd.DataFrame(rows)
