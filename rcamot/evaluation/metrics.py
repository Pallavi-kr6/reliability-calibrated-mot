"""HOTA / CLEAR (MOTA, IDSW) / Identity (IDF1) metrics, re-implemented after the TrackEval formulation
(Luiten et al., IJCV 2021; https://github.com/JonathonLuiten/TrackEval).

MOTChallenge pre-processing (simplified vs TrackEval): GT rows with flag==0 are dropped; only class 1
(pedestrian) is evaluated; tracker boxes that match (IoU>=0.5, Hungarian) a distractor-class GT box
(classes 2, 7, 8, 12) are removed. Use `main.py evaluate --backend trackeval` for the official numbers.
Metrics are returned in percent (0-100) except IDSW/TP/FN/FP counts.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..algorithm.geometry import iou_matrix
from ..datasets.mot_io import xywh_to_xyxy

EPS = np.finfo(float).eps
ALPHAS = np.arange(0.05, 0.9501, 0.05)
DISTRACTOR_CLASSES = (2, 7, 8, 12)


def _prepare(gt: np.ndarray, pred: np.ndarray, frame_range: Tuple[int, int], thr: float = 0.5):
    f0, f1 = frame_range
    gt = gt[(gt[:, 0] >= f0) & (gt[:, 0] <= f1) & (gt[:, 6] != 0)] if len(gt) else gt
    pred = pred[(pred[:, 0] >= f0) & (pred[:, 0] <= f1)] if len(pred) else np.zeros((0, 7))
    gt_by = _group(gt)
    pr_by = _group(pred)
    frames = []
    for f in range(f0, f1 + 1):
        g = gt_by.get(f, np.zeros((0, 9)))
        p = pr_by.get(f, np.zeros((0, 7)))
        is_ped = g[:, 7] == 1
        is_dis = np.isin(g[:, 7], DISTRACTOR_CLASSES)
        keep_tr = np.ones(len(p), bool)
        if len(p) and is_dis.any():
            sim_all = iou_matrix(xywh_to_xyxy(p[:, 2:6]), xywh_to_xyxy(g[:, 2:6]))
            r, c = linear_sum_assignment(-sim_all)
            for a, b in zip(r, c):
                if sim_all[a, b] >= thr - EPS and is_dis[b]:
                    keep_tr[a] = False
        g = g[is_ped]
        p = p[keep_tr]
        sim = iou_matrix(xywh_to_xyxy(g[:, 2:6]), xywh_to_xyxy(p[:, 2:6])) if len(g) and len(p) else np.zeros((len(g), len(p)))
        frames.append((f, g[:, 1].astype(int), p[:, 1].astype(int), sim))
    return frames


def _group(a: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    if len(a) == 0:
        return out
    order = np.argsort(a[:, 0], kind="stable")
    a = a[order]
    uniq, start = np.unique(a[:, 0], return_index=True)
    ends = np.r_[start[1:], len(a)]
    for f, s, e in zip(uniq, start, ends):
        out[int(f)] = a[s:e]
    return out


def evaluate(gt: np.ndarray, pred: np.ndarray, frame_range: Tuple[int, int], thr: float = 0.5,
             return_events: bool = False) -> Dict:
    frames = _prepare(np.asarray(gt, dtype=np.float64), np.asarray(pred, dtype=np.float64).reshape(-1, pred.shape[-1] if len(pred) else 7), frame_range, thr)
    gt_ids = np.unique(np.concatenate([fr[1] for fr in frames] + [np.zeros(0, int)]))
    tr_ids = np.unique(np.concatenate([fr[2] for fr in frames] + [np.zeros(0, int)]))
    G, T = len(gt_ids), len(tr_ids)
    gmap = {int(v): i for i, v in enumerate(gt_ids)}
    tmap = {int(v): i for i, v in enumerate(tr_ids)}
    fdata = [(f, np.array([gmap[int(x)] for x in g], int), np.array([tmap[int(x)] for x in p], int), s)
             for f, g, p, s in frames]
    num_gt = int(sum(len(x[1]) for x in fdata))
    num_pr = int(sum(len(x[2]) for x in fdata))

    # ---------------- first pass: id counts + potential matches (HOTA / Identity) ----------------
    gt_count, tr_count = np.zeros(G), np.zeros(T)
    potential = np.zeros((G, T))          # soft (HOTA global alignment)
    id_potential = np.zeros((G, T))       # hard IoU>=thr (Identity)
    for f, gi, ti, sim in fdata:
        gt_count[gi] += 1
        tr_count[ti] += 1
        if len(gi) and len(ti):
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            sim_iou = np.where(denom > EPS, sim / np.maximum(denom, EPS), 0.0)
            potential[gi[:, None], ti[None, :]] += sim_iou
            id_potential[gi[:, None], ti[None, :]] += (sim >= thr - EPS)
    global_align = potential / np.maximum(gt_count[:, None] + tr_count[None, :] - potential, EPS) if G and T else np.zeros((G, T))

    # ---------------- HOTA ----------------
    A = len(ALPHAS)
    h_tp, h_fn, h_fp, loc_sum = np.zeros(A), np.zeros(A), np.zeros(A), np.zeros(A)
    matches_count = [np.zeros((G, T)) for _ in range(A)]
    # ---------------- CLEAR ----------------
    c_tp = c_fn = c_fp = c_idsw = 0
    prev_tr = np.full(G, np.nan)
    prev_step_tr = np.full(G, np.nan)
    events: List[Tuple[int, int, int, int]] = []
    for f, gi, ti, sim in fdata:
        ng, nt = len(gi), len(ti)
        if ng == 0:
            h_fp += nt
            c_fp += nt
            prev_step_tr[:] = np.nan
            continue
        if nt == 0:
            h_fn += ng
            c_fn += ng
            prev_step_tr[:] = np.nan
            continue
        # HOTA matching
        score = global_align[gi[:, None], ti[None, :]] * sim
        r, c = linear_sum_assignment(-score)
        a_sim = sim[r, c]
        for a, alpha in enumerate(ALPHAS):
            m = a_sim >= alpha - EPS
            rr, cc = r[m], c[m]
            n = len(rr)
            if n:
                matches_count[a][gi[rr], ti[cc]] += 1
                loc_sum[a] += sim[rr, cc].sum()
            h_tp[a] += n
            h_fn[a] += ng - n
            h_fp[a] += nt - n
        # CLEAR matching (prefers continuing the previous assignment)
        bonus = (ti[None, :] == prev_step_tr[gi][:, None]).astype(float)
        sc = 1000.0 * bonus + sim
        sc[sim < thr - EPS] = 0.0
        r2, c2 = linear_sum_assignment(-sc)
        ok = sc[r2, c2] > EPS
        r2, c2 = r2[ok], c2[ok]
        mg, mt = gi[r2], ti[c2]
        prev = prev_tr[mg]
        sw = (~np.isnan(prev)) & (mt != prev)
        c_idsw += int(sw.sum())
        for k in np.where(sw)[0]:
            events.append((int(f), int(gt_ids[mg[k]]), int(tr_ids[int(prev[k])]), int(tr_ids[mt[k]])))
        prev_tr[mg] = mt
        prev_step_tr[:] = np.nan
        prev_step_tr[mg] = mt
        c_tp += len(r2)
        c_fn += ng - len(r2)
        c_fp += nt - len(r2)

    ass_num = np.zeros(A)
    for a in range(A):
        ass_a = matches_count[a] / np.maximum(1.0, gt_count[:, None] + tr_count[None, :] - matches_count[a]) if G and T else 0
        ass_num[a] = float((matches_count[a] * ass_a).sum()) if G and T else 0.0

    # ---------------- Identity (IDF1) ----------------
    idtp = idfn = idfp = 0.0
    if G + T > 0:
        fp_mat = np.zeros((G + T, G + T))
        fn_mat = np.zeros((G + T, G + T))
        fp_mat[G:, :T] = 1e10
        fn_mat[:G, T:] = 1e10
        fn_mat[:G, :T] = gt_count[:, None] - id_potential
        fp_mat[:G, :T] = tr_count[None, :] - id_potential
        fn_mat[np.arange(G), T + np.arange(G)] = gt_count
        fp_mat[G + np.arange(T), np.arange(T)] = tr_count
        r, c = linear_sum_assignment(fn_mat + fp_mat)
        idfn, idfp = float(fn_mat[r, c].sum()), float(fp_mat[r, c].sum())
        idtp = float(num_gt - idfn)

    raw = {"h_tp": h_tp, "h_fn": h_fn, "h_fp": h_fp, "loc_sum": loc_sum, "ass_num": ass_num,
           "c_tp": c_tp, "c_fn": c_fn, "c_fp": c_fp, "c_idsw": c_idsw, "idtp": idtp, "idfn": idfn, "idfp": idfp,
           "num_gt": num_gt, "num_pr": num_pr}
    res = _finalize(raw)
    if return_events:
        res["events"] = events
    res["_raw"] = raw
    return res


def _finalize(raw: Dict) -> Dict:
    tp, fn, fp = raw["h_tp"], raw["h_fn"], raw["h_fp"]
    det_a = tp / np.maximum(1.0, tp + fn + fp)
    ass_a = raw["ass_num"] / np.maximum(1.0, tp)
    loc_a = raw["loc_sum"] / np.maximum(1.0, tp)
    hota = np.sqrt(det_a * ass_a)
    idtp, idfn, idfp = raw["idtp"], raw["idfn"], raw["idfp"]
    n = max(raw["num_gt"], 1)
    return {"HOTA": float(100 * hota.mean()), "DetA": float(100 * det_a.mean()), "AssA": float(100 * ass_a.mean()), "LocA": float(100 * loc_a.mean()),
            "IDF1": 100 * idtp / max(idtp + 0.5 * idfn + 0.5 * idfp, EPS),
            "IDP": 100 * idtp / max(idtp + idfp, EPS), "IDR": 100 * idtp / max(idtp + idfn, EPS),
            "MOTA": 100 * (1 - (raw["c_fn"] + raw["c_fp"] + raw["c_idsw"]) / n), "IDSW": int(raw["c_idsw"]),
            "TP": int(raw["c_tp"]), "FN": int(raw["c_fn"]), "FP": int(raw["c_fp"]),
            "num_gt": int(raw["num_gt"]), "num_pred": int(raw["num_pr"])}


def combine(results: List[Dict]) -> Dict:
    """Combine per-sequence results (sum counts, TP-weighted AssA/LocA) like TrackEval's COMBINED row."""
    raws = [r["_raw"] for r in results]
    comb: Dict = {}
    for k in raws[0]:
        comb[k] = sum(r[k] for r in raws)
    out = _finalize(comb)
    out["_raw"] = comb
    return out


METRIC_KEYS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW")


def public(res: Dict) -> Dict:
    return {k: v for k, v in res.items() if not k.startswith("_") and k != "events"}
