"""Track object, lifecycle states and EMA appearance prototype."""
from __future__ import annotations

from enum import IntEnum
from typing import Optional

import numpy as np

from .geometry import xyah_to_xyxy, xyxy_to_xyah
from .kalman import KalmanFilter


class TrackState(IntEnum):
    TENTATIVE = 1
    TRACKED = 2
    LOST = 3
    REMOVED = 4


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


class Track:
    """One identity hypothesis: Kalman state + EMA appearance prototype + lifecycle."""

    def __init__(self, tid: int, box: np.ndarray, score: float, emb: Optional[np.ndarray],
                 frame: int, kf: KalmanFilter, state: TrackState = TrackState.TENTATIVE,
                 vel_decay: float = 0.9):
        self.id = tid
        self.kf = kf
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(np.asarray(box, dtype=np.float64)))
        self.proto = _unit(np.asarray(emb, dtype=np.float64)) if emb is not None else None
        self.score = float(score)
        self.hits = 1
        self.age = 1
        self.start_frame = frame
        self.last_frame = frame  # last frame with an associated observation
        self.state = state
        self.vel_decay = vel_decay

    # ---- motion ----
    def predict(self, frame: int) -> None:
        if frame - self.last_frame > 1:  # lost: damp velocity so the prediction does not fly away
            self.mean[4:8] *= self.vel_decay
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1

    @property
    def pred_box(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])

    def tsu(self, frame: int) -> int:
        """Frames since the last associated observation (>=1 for a track seen last frame)."""
        return max(1, frame - self.last_frame)

    # ---- update ----
    def update(self, box: np.ndarray, score: float, emb: Optional[np.ndarray], frame: int,
               alpha: float, min_hits: int = 2) -> None:
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(np.asarray(box, dtype=np.float64)))
        self.score = float(score)
        self.hits += 1
        self.last_frame = frame
        if emb is not None:
            e = np.asarray(emb, dtype=np.float64)
            self.proto = _unit(e) if self.proto is None else _unit(alpha * self.proto + (1.0 - alpha) * e)
        if self.state == TrackState.LOST or (self.state == TrackState.TENTATIVE and self.hits >= min_hits):
            self.state = TrackState.TRACKED

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED
