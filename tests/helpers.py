"""Scripted in-memory scene: two people with SIMILAR appearance cross (with overlap/occlusion), then person A
disappears for a few frames and reappears."""
import numpy as np

from rcamot.utils import load_config

N_FRAMES = 80
GAP = (45, 55)  # A absent for frames 45..54


def unit(v):
    return v / np.linalg.norm(v)


def scripted_scene(seed=0, dim=16, similarity=0.85):
    rng = np.random.default_rng(seed)
    ea = unit(rng.normal(size=dim))
    perp = unit(rng.normal(size=dim) - ea * (ea @ rng.normal(size=dim)))
    perp = unit(perp - ea * (ea @ perp))
    eb = unit(similarity * ea + np.sqrt(1 - similarity ** 2) * perp)
    frames, gt = [], []
    for t in range(1, N_FRAMES + 1):
        xa = 40 + 4.0 * t
        xb = 40 + 4.0 * 60 - 4.0 * t - 0  # B moves the other way; they meet near t=30
        boxes, embs, ids = [], [], []
        overlap = abs(t - 30) <= 3
        if not (GAP[0] <= t < GAP[1]):
            boxes.append([xa, 100, xa + 40, 200])
            embs.append(unit((0.5 * ea + 0.5 * eb) if overlap else ea) if False else unit((0.55 * ea + 0.45 * eb) if overlap else ea + rng.normal(0, 0.03, dim)))
            ids.append(1)
        boxes.append([xb, 102, xb + 40, 202])
        embs.append(unit((0.45 * ea + 0.55 * eb) if overlap else eb + rng.normal(0, 0.03, dim)))
        ids.append(2)
        boxes = np.array(boxes, float) + rng.normal(0, 0.4, (len(boxes), 4))
        frames.append((t, boxes, np.full(len(boxes), 0.9), np.array(embs), np.array(ids)))
        for b, i in zip(boxes, ids):
            gt.append([t, i, b[0], b[1], b[2] - b[0], b[3] - b[1], 1, 1, 1.0])
    return frames, np.array(gt)


def run_scene(method, cfg_over=None, model=None):
    from rcamot.algorithm.tracker import MOTTracker
    cfg = load_config("configs/default.yaml", dataset="synthetic", overrides={"method": method, **(cfg_over or {})})
    tr = MOTTracker(cfg, model)
    rows = []
    for t, b, s, e, _ in scripted_scene()[0]:
        for (tid, x1, y1, x2, y2, sc) in tr.update(t, b, s, e):
            rows.append([t, tid, x1, y1, x2 - x1, y2 - y1, sc])
    return np.array(rows), scripted_scene()[1]
