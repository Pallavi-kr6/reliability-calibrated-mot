"""Detection + embedding cache.

Layout: <cache_dir>/<dataset>/<seq>/<det_source>__<embedder>__f<start>-<end>.npz  (+ .json fingerprint)
The fingerprint holds every input that influences the content (det source/params, embedder, frame range, seed,
source-file size+mtime, cache version). A mismatch triggers an automatic recompute; `--force` always recomputes.
All tracker configurations (B0..B3, RCA, ablations, stress) reuse the same cached arrays.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..algorithm.geometry import iou_matrix
from ..datasets.mot_io import load_det, load_gt, xywh_to_xyxy
from ..datasets.registry import SequenceSpec
from ..utils import config_hash, ensure_dir, get_logger, resolve_path
from .embedder import Embedder

log = get_logger()
CACHE_VERSION = 1


@dataclass
class DetSet:
    spec: SequenceSpec
    frames: np.ndarray        # (N,) int
    boxes: np.ndarray         # (N,4) xyxy
    scores: np.ndarray        # (N,)
    gt_id: np.ndarray         # (N,) matched GT identity or -1
    vis: np.ndarray           # (N,) visibility of the matched GT (1 if unmatched)
    emb: Optional[np.ndarray]  # (N,D) float32 or None

    def frame_slices(self) -> Dict[int, slice]:
        out: Dict[int, slice] = {}
        if len(self.frames) == 0:
            return out
        order = np.argsort(self.frames, kind="stable")
        assert np.all(order == np.arange(len(order))), "DetSet must be sorted by frame"
        uniq, start = np.unique(self.frames, return_index=True)
        ends = np.r_[start[1:], len(self.frames)]
        for f, s, e in zip(uniq, start, ends):
            out[int(f)] = slice(int(s), int(e))
        return out

    def drop(self, mask: np.ndarray) -> "DetSet":
        keep = ~np.asarray(mask, dtype=bool)
        return DetSet(self.spec, self.frames[keep], self.boxes[keep], self.scores[keep], self.gt_id[keep],
                      self.vis[keep], None if self.emb is None else self.emb[keep])


def gt_eval_rows(spec: SequenceSpec) -> np.ndarray:
    """GT rows inside the sequence's frame range (all classes/flags; metrics decide what to use)."""
    g = load_gt(spec.gt_path)
    return g[(g[:, 0] >= spec.frame_start) & (g[:, 0] <= spec.frame_end)]


def _ped_gt(spec: SequenceSpec) -> np.ndarray:
    g = gt_eval_rows(spec)
    return g[(g[:, 6] != 0) & (g[:, 7] == 1)]


def _oracle_dets(spec: SequenceSpec, oc: Dict, seed: int):
    rng = np.random.default_rng(seed)
    g = _ped_gt(spec)
    g = g[g[:, 8] >= oc["min_vis"]]
    keep = rng.random(len(g)) >= oc["miss_prob"]
    g = g[keep]
    box = xywh_to_xyxy(g[:, 2:6])
    wh = np.stack([box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]], 1)
    jit = rng.normal(0, oc["jitter_std"], (len(g), 4)) * np.tile(wh, 2)
    box = box + jit
    score = np.clip(rng.normal(oc["score_mean"], oc["score_std"], len(g)), 0.2, 0.999)
    return g[:, 0].astype(int), box, score, g[:, 1].astype(int), g[:, 8]


def _match_to_gt(spec: SequenceSpec, frames, boxes):
    """Label detections with GT identity (IoU>=0.5, Hungarian per frame)."""
    g = _ped_gt(spec)
    gt_id = np.full(len(frames), -1, int)
    vis = np.ones(len(frames))
    gby = {int(f): g[g[:, 0] == f] for f in np.unique(g[:, 0])}
    for f in np.unique(frames):
        idx = np.where(frames == f)[0]
        rows = gby.get(int(f))
        if rows is None or len(rows) == 0:
            continue
        iou = iou_matrix(boxes[idx], xywh_to_xyxy(rows[:, 2:6]))
        r, c = linear_sum_assignment(-iou)
        for a, b in zip(r, c):
            if iou[a, b] >= 0.5:
                gt_id[idx[a]] = int(rows[b, 1])
                vis[idx[a]] = rows[b, 8]
    return gt_id, vis


def _fingerprint(spec: SequenceSpec, data_cfg: Dict, embedder: Optional[Embedder], seed: int) -> Dict:
    src = None
    if data_cfg["det_source"] in ("public", "file"):
        p = resolve_path(data_cfg["det_file"].format(seq_dir=spec.dir)) if data_cfg["det_source"] == "file" else spec.det_path
        if p.exists():
            st = p.stat()
            src = [str(p.name), st.st_size]
    return {"version": CACHE_VERSION, "dataset": spec.dataset, "seq": spec.name, "range": [spec.frame_start, spec.frame_end],
            "det_source": data_cfg["det_source"], "oracle": data_cfg["oracle"] if data_cfg["det_source"] == "oracle" else None,
            "min_score": data_cfg["det_min_score"], "embedder": embedder.name if embedder else None,
            "embedder_model": data_cfg.get("embedder_model") if embedder and embedder.name == "osnet" else None,
            "seed": seed, "source": src}


def cache_path(cache_dir, spec: SequenceSpec, data_cfg: Dict):
    emb = data_cfg["embedder"] if data_cfg.get("embedder") else "none"
    return ensure_dir(resolve_path(cache_dir) / spec.dataset / spec.name) / \
        f"{data_cfg['det_source']}__{emb}__f{spec.frame_start}-{spec.frame_end}.npz"


def build_detset(spec: SequenceSpec, data_cfg: Dict, embedder: Optional[Embedder], cache_dir: str,
                 seed: int = 0, force: bool = False) -> DetSet:
    path = cache_path(cache_dir, spec, data_cfg)
    meta = path.with_suffix(".json")
    fp = _fingerprint(spec, data_cfg, embedder, seed)
    if path.exists() and meta.exists() and not force:
        if json.loads(meta.read_text()) == json.loads(json.dumps(fp, default=str)):
            z = np.load(path)
            return DetSet(spec, z["frames"], z["boxes"], z["scores"], z["gt_id"], z["vis"], z["emb"] if "emb" in z else None)
        log.info(f"[cache] fingerprint changed for {spec.name}: recomputing")

    src = data_cfg["det_source"]
    if src == "oracle":
        frames, boxes, scores, gt_id, vis = _oracle_dets(spec, data_cfg["oracle"], seed)
    elif src in ("public", "file"):
        p = spec.det_path if src == "public" else resolve_path(data_cfg["det_file"].format(seq_dir=spec.dir))
        if not p.exists():
            raise FileNotFoundError(f"Detection file not found: {p}. Use data.det_source=oracle (D0) or supply det files "
                                    f"(DanceTrack ships none; run any detector and write MOT-format det.txt).")
        d = load_det(p)
        d = d[(d[:, 0] >= spec.frame_start) & (d[:, 0] <= spec.frame_end) & (d[:, 6] >= data_cfg["det_min_score"])]
        frames, boxes, scores = d[:, 0].astype(int), xywh_to_xyxy(d[:, 2:6]), d[:, 6]
        gt_id, vis = _match_to_gt(spec, frames, boxes)
    else:
        raise ValueError(f"Unknown det_source '{src}'")
    boxes = boxes.copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, max(spec.width - 1, 1))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, max(spec.height - 1, 1))
    ok = ((boxes[:, 2] - boxes[:, 0]) > 2) & ((boxes[:, 3] - boxes[:, 1]) > 2)
    order = np.argsort(frames[ok], kind="stable")
    frames, boxes, scores, gt_id, vis = (a[ok][order] for a in (frames, boxes, scores, gt_id, vis))

    emb = None
    if embedder is not None:
        emb = np.zeros((len(frames), embedder.dim), np.float32)
        sl = DetSet(spec, frames, boxes, scores, gt_id, vis, None).frame_slices()
        for k, (f, s) in enumerate(sl.items()):
            img = cv2.imread(str(spec.image_path(f)))
            if img is None:
                raise FileNotFoundError(f"Image not found: {spec.image_path(f)}")
            crops = []
            for b in boxes[s]:
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(np.ceil(b[2])), int(np.ceil(b[3]))
                crops.append(img[y1:y2, x1:x2])
            emb[s] = embedder.embed(crops)
            if k % 200 == 0:
                log.info(f"[cache] {spec.name}: embedded frame {f} ({k + 1}/{len(sl)})")
    arrays = dict(frames=frames, boxes=boxes.astype(np.float32), scores=scores.astype(np.float32), gt_id=gt_id, vis=vis.astype(np.float32))
    if emb is not None:
        arrays["emb"] = emb
    np.savez_compressed(path, **arrays)
    meta.write_text(json.dumps(fp, default=str))
    log.info(f"[cache] wrote {path.relative_to(resolve_path('.')) if str(path).startswith(str(resolve_path('.'))) else path} ({len(frames)} detections)")
    return DetSet(spec, frames, boxes.astype(np.float32), scores.astype(np.float32), gt_id, vis.astype(np.float32), emb)
