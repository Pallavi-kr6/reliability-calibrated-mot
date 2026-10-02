"""Oracle-track pair mining for RCA.

Labels must not inherit a baseline tracker's identity drift, so every GT identity gets an *oracle* track
(Kalman + EMA prototype) that is updated only with its true detection. At every frame we form all
(oracle track, high-score detection) pairs inside the loose gate with y = [gt_id(det) == track identity],
computing the SAME features the online tracker uses (features.compute_pair_features).
Known limitation: the oracle state distribution differs slightly from an online tracker's (see docs/methodology.md).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..algorithm.features import FEATURE_KEYS, compute_pair_features, track_occlusion
from ..algorithm.geometry import max_other_ioa
from ..algorithm.kalman import KalmanFilter
from ..algorithm.track import Track, TrackState
from ..inference.cache import DetSet


def mine_pairs(ds: DetSet, cfg: Dict) -> Dict[str, np.ndarray]:
    if ds.emb is None:
        raise ValueError("Pair mining needs embeddings (cache them with an embedder first).")
    T, G = cfg["tracker"], cfg["gate"]
    kf = KalmanFilter()
    oracle: Dict[int, Track] = {}
    sl = ds.frame_slices()
    out: Dict[str, List[np.ndarray]] = {k: [] for k in FEATURE_KEYS + ("label", "frame", "block")}
    block = int(cfg["rca"]["fit_block"])
    for f in ds.spec.frames:
        for tr in oracle.values():
            tr.predict(f)
        s = sl.get(f)
        if s is None:
            continue
        boxes, scores, embs, gid = ds.boxes[s], ds.scores[s], ds.emb[s], ds.gt_id[s]
        high = np.where(scores >= T["high_thresh"])[0]
        alive = [tr for tr in oracle.values() if f - tr.last_frame <= T["max_age"]]
        if alive and len(high):
            pred = np.stack([t.pred_box for t in alive])
            tsu = np.array([t.tsu(f) for t in alive])
            protos = np.stack([t.proto for t in alive])
            fe = compute_pair_features(pred, tsu, protos, boxes[high], scores[high], embs[high],
                                       max_other_ioa(boxes)[high], track_occlusion(pred, tsu, T["occ_track_max_tsu"]),
                                       G, cfg["rca"]["margin_clip"])
            tid = np.array([t.id for t in alive])
            label = (gid[high][None, :] == tid[:, None])
            sel = fe.mask
            for k, v in fe.as_dict().items():
                out[k].append(v[sel])
            out["label"].append(label[sel].astype(np.float64))
            out["frame"].append(np.full(int(sel.sum()), f))
            out["block"].append(np.full(int(sel.sum()), (f - ds.spec.frame_start) // block))
        # update oracle tracks with their true detections
        for g in np.unique(gid[gid >= 0]):
            cand = np.where((gid == g) & (scores >= T["low_thresh"]))[0]
            if len(cand) == 0:
                continue
            j = cand[np.argmax(scores[cand])]
            if g in oracle and f - oracle[g].last_frame <= T["max_age"]:
                oracle[g].update(boxes[j], scores[j], embs[j], f, T["ema_alpha"], T["min_hits"])
            else:
                oracle[g] = Track(int(g), boxes[j], scores[j], embs[j], f, kf, TrackState.TRACKED, T["lost_vel_decay"])
        for g in [g for g, tr in oracle.items() if f - tr.last_frame > T["max_age"]]:
            del oracle[g]
    if not out["label"]:
        return {k: np.zeros(0) for k in out}
    return {k: np.concatenate(v) for k, v in out.items()}


def concat_pairs(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    parts = [p for p in parts if len(p.get("label", [])) > 0]
    if not parts:
        raise ValueError("No training pairs were mined (empty/invalid detections?).")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
