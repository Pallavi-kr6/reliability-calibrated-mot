"""Glue: build cached detection sets, run a tracker over a sequence, load/auto-train the RCA model."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..algorithm.tracker import MOTTracker
from ..datasets.registry import SequenceSpec, list_sequences
from ..models.rca import RCAModel
from ..utils import get_logger, resolve_path
from .cache import DetSet, build_detset
from .embedder import get_embedder

log = get_logger()


def get_specs(cfg: Dict, split: str, only: Optional[List[str]] = None) -> List[SequenceSpec]:
    d = cfg["data"]
    return list_sequences(d["dataset"], split, d["root"], d.get("mot17_detector", "FRCNN"), only)


def get_detsets(cfg: Dict, split: str, force: bool = False, only: Optional[List[str]] = None) -> List[DetSet]:
    specs = get_specs(cfg, split, only)
    emb = get_embedder(cfg["data"], cfg.get("device", "auto"))
    return [build_detset(s, cfg["data"], emb, cfg["cache_dir"], cfg["seed"], force) for s in specs]


def run_tracker(ds: DetSet, cfg: Dict, model: Optional[RCAModel] = None, collect_debug: bool = False):
    """Run the online tracker over one DetSet. Returns (rows [frame,id,x,y,w,h,score], stats)."""
    tr = MOTTracker(cfg, model)
    sl = ds.frame_slices()
    rows, debug, per_frame = [], {}, []
    import time
    for f in ds.spec.frames:
        s = sl.get(f)
        if s is None:
            b, sc, e = np.zeros((0, 4)), np.zeros(0), None
        else:
            b, sc, e = ds.boxes[s], ds.scores[s], (ds.emb[s] if ds.emb is not None else None)
        t0 = time.perf_counter()
        out = tr.update(f, b, sc, e)
        per_frame.append((len(sc), time.perf_counter() - t0))
        for (tid, x1, y1, x2, y2, score) in out:
            rows.append([f, tid, x1, y1, x2 - x1, y2 - y1, score])
        if collect_debug and tr.last_debug:
            debug[f] = list(tr.last_debug)
    arr = np.array(rows) if rows else np.zeros((0, 7))
    stats = {"timers": dict(tr.timers), "per_frame": per_frame, "n_frames": len(per_frame), "debug": debug}
    return arr, stats


def ensure_model(cfg: Dict, log_prefix: str = "") -> RCAModel:
    path = resolve_path(cfg["rca"]["model_path"])
    if path.exists():
        return RCAModel.load(path)
    if not cfg["rca"].get("auto_train", False):
        raise FileNotFoundError(f"RCA parameters missing: {path}. Run: python main.py train --config configs/rca.yaml")
    log.info(f"{log_prefix}RCA parameters missing -> training now (rca.auto_train=true)")
    from ..training.train_rca import train_rca
    model, _ = train_rca(cfg)
    return model
