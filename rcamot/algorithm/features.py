"""Pair features shared by the online tracker AND the offline pair miner (no train/serve skew).

For track i (predicted box, prototype, frames-since-update) and detection j:
  iou      IoU(predicted box, detection)
  ctr      centre distance / mean box diagonal
  tsu      log(frames since last association)   (0 for a track seen last frame)
  d        cosine distance between prototype and detection embedding
  occ      max fraction of either box covered by another object  (occlusion / crowding)
  margin   candidate-set margin: (best competing appearance distance among gated candidates) - d
  score    detection confidence
  mask     loose spatial gate (candidate set)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .geometry import center_dist_matrix, iou_matrix, max_other_ioa

FEATURE_KEYS = ("iou", "ctr", "tsu", "d", "occ", "margin", "score")


@dataclass
class PairFeatures:
    iou: np.ndarray
    ctr: np.ndarray
    tsu: np.ndarray
    d: np.ndarray
    occ: np.ndarray
    margin: np.ndarray
    score: np.ndarray
    mask: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in FEATURE_KEYS}

    @property
    def shape(self):
        return self.iou.shape


def candidate_margin(d: np.ndarray, mask: np.ndarray, clip: float = 0.3) -> np.ndarray:
    """Candidate-set appearance margin.

    m_ij = min( min_{j'!=j, gated} d_ij' , min_{i'!=i, gated} d_i'j ) - d_ij, clipped to [-clip, clip].
    Large positive: this pair is distinctly closer in appearance than every competitor.
    ~0 or negative: appearance cannot separate the competing candidates (similar clothes / crowding).
    No competitor at all -> +clip (appearance is unambiguous by construction).
    """
    n, m = d.shape
    if n == 0 or m == 0:
        return np.zeros((n, m))
    dm = np.where(mask, d, np.inf)

    def competitor_min(x: np.ndarray) -> np.ndarray:
        # for each (i, j): min over the *other* columns of row i  (axis=1)
        if x.shape[1] == 1:
            return np.full(x.shape, np.inf)
        idx = np.argmin(x, axis=1)
        min1 = x[np.arange(x.shape[0]), idx]
        tmp = x.copy()
        tmp[np.arange(x.shape[0]), idx] = np.inf
        min2 = tmp.min(axis=1)
        out = np.repeat(min1[:, None], x.shape[1], axis=1)
        out[np.arange(x.shape[0]), idx] = min2
        return out

    comp_row = competitor_min(dm)          # other detections competing for track i
    comp_col = competitor_min(dm.T).T      # other tracks competing for detection j
    comp = np.minimum(comp_row, comp_col)
    margin = np.where(np.isfinite(comp), comp - d, clip)
    return np.clip(margin, -clip, clip)


def gate_mask(iou: np.ndarray, ctr: np.ndarray, tsu_frames: np.ndarray, gate: Dict) -> np.ndarray:
    """Loose spatial gate: IoU>gate_iou OR centre distance below a radius that grows with time lost."""
    radius = np.minimum(gate.get("ctr", 0.75) * (1.0 + gate.get("growth", 0.1) * (tsu_frames[:, None] - 1)),
                        gate.get("ctr_max", 2.5))
    return (iou > gate.get("iou", 0.0)) | (ctr < radius)


def compute_pair_features(pred_boxes: np.ndarray, tsu_frames: np.ndarray, protos: Optional[np.ndarray],
                          det_boxes: np.ndarray, det_scores: np.ndarray, det_embs: Optional[np.ndarray],
                          det_occ: np.ndarray, trk_occ: np.ndarray, gate: Dict,
                          margin_clip: float = 0.3) -> PairFeatures:
    n, m = len(pred_boxes), len(det_boxes)
    tsu_frames = np.asarray(tsu_frames, dtype=np.float64).reshape(-1)
    iou = iou_matrix(pred_boxes, det_boxes)
    ctr = center_dist_matrix(pred_boxes, det_boxes)
    if n and m and protos is not None and det_embs is not None:
        d = np.clip(1.0 - np.asarray(protos) @ np.asarray(det_embs).T, 0.0, 2.0)
    else:
        d = np.ones((n, m))
    mask = gate_mask(iou, ctr, tsu_frames, gate) if n and m else np.zeros((n, m), dtype=bool)
    margin = candidate_margin(d, mask, margin_clip)
    tsu_f = np.log(np.maximum(tsu_frames, 1.0))[:, None] * np.ones((1, m))
    occ = np.maximum(np.asarray(trk_occ, dtype=np.float64).reshape(-1, 1),
                     np.asarray(det_occ, dtype=np.float64).reshape(1, -1)) if n and m else np.zeros((n, m))
    score = np.ones((n, 1)) * np.asarray(det_scores, dtype=np.float64).reshape(1, -1) if n and m else np.zeros((n, m))
    return PairFeatures(iou, ctr, tsu_f, d, occ, margin, score, mask)


def track_occlusion(pred_boxes: np.ndarray, tsu_frames: np.ndarray, max_tsu: int = 5) -> np.ndarray:
    """Per-track occlusion from other *recently seen* tracks' predicted boxes."""
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    if len(pred_boxes) < 2:
        return np.zeros(len(pred_boxes))
    recent = np.asarray(tsu_frames) <= max_tsu
    occ = np.zeros(len(pred_boxes))
    idx = np.where(recent)[0]
    if len(idx) >= 2:
        from .geometry import ioa_matrix
        m = ioa_matrix(pred_boxes, pred_boxes[idx])
        for k, i in enumerate(idx):
            m[i, k] = 0.0
        occ = m.max(axis=1)
    return occ
