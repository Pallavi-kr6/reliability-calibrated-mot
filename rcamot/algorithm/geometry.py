"""Box geometry. All boxes are float arrays in xyxy format unless stated otherwise."""
from __future__ import annotations

import numpy as np


def box_area(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def _inter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    return wh[..., 0] * wh[..., 1]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4) -> (N,M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    inter = _inter(a, b)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def ioa_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over area of `a`: fraction of a[i] covered by b[j] -> (N,M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    inter = _inter(a, b)
    area = box_area(a)[:, None]
    return np.where(area > 0, inter / np.maximum(area, 1e-12), 0.0)


def max_other_ioa(boxes: np.ndarray) -> np.ndarray:
    """For each box, the max fraction of its area covered by any *other* box in the set."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    n = len(boxes)
    if n < 2:
        return np.zeros(n)
    m = ioa_matrix(boxes, boxes)
    np.fill_diagonal(m, 0.0)
    return m.max(axis=1)


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64)
    w = b[..., 2] - b[..., 0]
    h = b[..., 3] - b[..., 1]
    return np.stack([b[..., 0] + w / 2, b[..., 1] + h / 2, w / np.maximum(h, 1e-6), h], axis=-1)


def xyah_to_xyxy(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    h = s[..., 3]
    w = s[..., 2] * h
    return np.stack([s[..., 0] - w / 2, s[..., 1] - h / 2, s[..., 0] + w / 2, s[..., 1] + h / 2], axis=-1)


def center_dist_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Center distance normalised by the mean box diagonal of each pair -> (N,M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ca = (a[:, :2] + a[:, 2:]) / 2
    cb = (b[:, :2] + b[:, 2:]) / 2
    da = np.hypot(a[:, 2] - a[:, 0], a[:, 3] - a[:, 1])
    db = np.hypot(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
    dist = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
    return dist / np.maximum((da[:, None] + db[None, :]) / 2, 1e-6)
