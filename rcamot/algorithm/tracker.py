"""Online (causal) multi-object tracker with a ByteTrack-style two-stage association.

method: b0 | b1 | b2 | b3 | rca   (see association.py). All share the same skeleton: Kalman prediction,
stage-1 (high-score detections vs confirmed+lost tracks), tentative-track matching, stage-2 (low-score
detections vs still-tracked tracks, IoU only), lifecycle management and EMA appearance prototypes.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from ..models.rca import RCAModel
from .association import assign, fused_cost, iou_cost, rca_cost
from .features import compute_pair_features, track_occlusion
from .geometry import iou_matrix, max_other_ioa
from .kalman import KalmanFilter
from .track import Track, TrackState

METHODS = ("b0", "b1", "b2", "b3", "rca")


class MOTTracker:
    def __init__(self, cfg: Dict, model: Optional[RCAModel] = None):
        self.cfg = cfg
        self.t = cfg["tracker"]
        self.method = cfg["method"]
        if self.method not in METHODS:
            raise ValueError(f"Unknown method '{self.method}', expected one of {METHODS}")
        if self.method == "rca" and model is None:
            raise ValueError("method 'rca' needs a fitted RCAModel (run `python main.py train`).")
        self.model = model
        self.kf = KalmanFilter()
        self.reset()

    def reset(self) -> None:
        self.tracks: List[Track] = []
        self.next_id = 1
        self.first_frame: Optional[int] = None
        self.timers = {"features": 0.0, "assign": 0.0, "update": 0.0, "total": 0.0}
        self.last_debug: List[Dict] = []

    # ------------------------------------------------------------------
    def _alpha(self, base: float, score: float, post: Optional[float], occ: float) -> float:
        mode = self.t["update_mode"]
        if mode == "score":
            sig = self.t["high_thresh"]
            conf = float(np.clip((score - sig) / max(1.0 - sig, 1e-6), 0.0, 1.0))
            return base + (1.0 - base) * (1.0 - conf)
        if mode == "posterior" and post is not None:
            return base + (1.0 - base) * (1.0 - post)
        if mode == "gated" and occ > self.cfg["fusion"]["ioa_gate"]:
            return 1.0  # skip the appearance update (prototype unchanged)
        return base

    def update(self, frame: int, boxes: np.ndarray, scores: np.ndarray, embs: Optional[np.ndarray]) -> List[tuple]:
        """Process one frame. Returns [(track_id, x1, y1, x2, y2, score), ...] for tracks observed this frame."""
        t_start = time.perf_counter()
        T = self.t
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        first = self.first_frame is None
        if first:
            self.first_frame = frame
        self.last_debug = []

        for tr in self.tracks:
            tr.predict(frame)

        high = np.where(scores >= T["high_thresh"])[0]
        low = np.where((scores >= T["low_thresh"]) & (scores < T["high_thresh"]))[0]
        confirmed = [tr for tr in self.tracks if tr.state in (TrackState.TRACKED, TrackState.LOST)]
        tentative = [tr for tr in self.tracks if tr.state == TrackState.TENTATIVE]
        use_app = embs is not None and self.method != "b0"
        matched_det, matched_trk = set(), set()

        # ---------------- stage 1 ----------------
        t0 = time.perf_counter()
        if confirmed and len(high):
            pred = np.stack([tr.pred_box for tr in confirmed])
            tsu = np.array([tr.tsu(frame) for tr in confirmed])
            protos = np.stack([tr.proto for tr in confirmed]) if use_app and all(tr.proto is not None for tr in confirmed) else None
            det_occ_all = max_other_ioa(boxes)
            trk_occ = track_occlusion(np.stack([tr.pred_box for tr in self.tracks]),
                                      np.array([tr.tsu(frame) for tr in self.tracks]), T["occ_track_max_tsu"])
            trk_occ_map = {id(tr): trk_occ[k] for k, tr in enumerate(self.tracks)}
            trk_occ_c = np.array([trk_occ_map[id(tr)] for tr in confirmed])
            f = compute_pair_features(pred, tsu, protos, boxes[high], scores[high],
                                      embs[high] if use_app else None, det_occ_all[high], trk_occ_c,
                                      self.cfg["gate"], self.cfg["rca"]["margin_clip"])
            t1 = time.perf_counter()
            if self.method == "rca":
                cost, valid, info = rca_cost(f, self.model, self.cfg["rca"])
            elif self.method == "b0":
                cost, valid = iou_cost(f.iou, self.cfg["fusion"]["b0_min_iou"])
                valid &= f.mask
                info = {}
            else:
                cost, valid, info = fused_cost(f, self.method, self.cfg["fusion"], T["high_thresh"])
            matches, _, _ = assign(cost, valid)
            t2 = time.perf_counter()
            self.timers["features"] += t1 - t0
            self.timers["assign"] += t2 - t1
            for i, j in matches:
                tr, dj = confirmed[i], int(high[j])
                post = float(info["post"][i, j]) if "post" in info else None
                alpha = self._alpha(T["ema_alpha"], scores[dj], post, float(f.occ[i, j]))
                tr.update(boxes[dj], scores[dj], embs[dj] if embs is not None else None, frame, alpha, T["min_hits"])
                matched_det.add(dj)
                matched_trk.add(id(tr))
                self.last_debug.append({"id": tr.id, "det": dj, "post": post, "iou": float(f.iou[i, j]),
                                        "d": float(f.d[i, j]), "margin": float(f.margin[i, j]),
                                        "occ": float(f.occ[i, j]),
                                        "lam": float(info["lam"][i, j]) if "lam" in info else None})

        # ---------------- tentative tracks vs remaining high detections ----------------
        t3 = time.perf_counter()
        rem_high = [int(j) for j in high if int(j) not in matched_det]
        if tentative and rem_high:
            pred = np.stack([tr.pred_box for tr in tentative])
            iou = iou_matrix(pred, boxes[rem_high])
            cost, valid = iou_cost(iou, T["tentative_min_iou"])
            for i, j in assign(cost, valid)[0]:
                tr, dj = tentative[i], rem_high[j]
                tr.update(boxes[dj], scores[dj], embs[dj] if embs is not None else None, frame, T["ema_alpha"], T["min_hits"])
                matched_det.add(dj)
                matched_trk.add(id(tr))

        # ---------------- stage 2: low-score detections, IoU only ----------------
        rem_trk = [tr for tr in confirmed if id(tr) not in matched_trk and tr.state == TrackState.TRACKED]
        rem_low = [int(j) for j in low if int(j) not in matched_det]
        if rem_trk and rem_low:
            pred = np.stack([tr.pred_box for tr in rem_trk])
            iou = iou_matrix(pred, boxes[rem_low])
            cost, valid = iou_cost(iou, T["stage2_min_iou"])
            for i, j in assign(cost, valid)[0]:
                tr, dj = rem_trk[i], rem_low[j]
                tr.update(boxes[dj], scores[dj], None, frame, T["ema_alpha"], T["min_hits"])  # no appearance from weak boxes
                matched_det.add(dj)
                matched_trk.add(id(tr))

        # ---------------- lifecycle ----------------
        for tr in self.tracks:
            if id(tr) in matched_trk:
                continue
            if tr.state == TrackState.TENTATIVE:
                tr.mark_removed()
            elif tr.state == TrackState.TRACKED:
                tr.mark_lost()
            if tr.state == TrackState.LOST and frame - tr.last_frame > T["max_age"]:
                tr.mark_removed()

        for dj in high:
            dj = int(dj)
            if dj in matched_det or scores[dj] < T["new_track_thresh"]:
                continue
            state = TrackState.TRACKED if first else TrackState.TENTATIVE
            self.tracks.append(Track(self.next_id, boxes[dj], scores[dj], embs[dj] if embs is not None else None,
                                     frame, self.kf, state, T["lost_vel_decay"]))
            self.next_id += 1
        self.tracks = [tr for tr in self.tracks if tr.state != TrackState.REMOVED]

        out = [(tr.id, *tr.pred_box.tolist(), tr.score) for tr in self.tracks
               if tr.state == TrackState.TRACKED and tr.last_frame == frame]
        self.timers["update"] += time.perf_counter() - t3
        self.timers["total"] += time.perf_counter() - t_start
        return out
