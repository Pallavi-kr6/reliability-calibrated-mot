"""Run a configured tracker over a split, write MOT result files + metrics.json."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..datasets.mot_io import read_mot_txt, write_mot_txt
from ..inference.cache import DetSet, gt_eval_rows
from ..inference.pipeline import ensure_model, get_detsets, run_tracker
from ..utils import ensure_dir, get_logger, load_config, resolve_path, save_json, set_seed
from .metrics import combine, evaluate, public

log = get_logger()

METHOD_CONFIGS = {"b0": "configs/baseline_b0.yaml", "b1": "configs/baseline_b1.yaml", "b2": "configs/baseline_b2.yaml",
                  "b3": "configs/baseline_b3.yaml", "rca": "configs/rca.yaml"}


def method_cfg(method: str, dataset: Optional[str], overrides: Optional[Dict] = None) -> Dict:
    if method not in METHOD_CONFIGS:
        raise ValueError(f"Unknown method {method}; choose from {list(METHOD_CONFIGS)}")
    return load_config(METHOD_CONFIGS[method], dataset=dataset, overrides=overrides)


def run_dir(cfg: Dict, name: Optional[str] = None) -> Path:
    return resolve_path(cfg["results_dir"]) / "runs" / str(cfg["data"]["dataset"]) / (name or cfg["name"])


def read_predictions(rdir: Path, detsets: List[DetSet]) -> Dict[str, np.ndarray]:
    out = {}
    for ds in detsets:
        p = rdir / f"{ds.spec.name}.txt"
        if not p.exists():
            raise FileNotFoundError(f"Missing result file {p}. Run `python main.py track --config <cfg>` first.")
        a = read_mot_txt(p)
        out[ds.spec.name] = a if len(a) else np.zeros((0, 7))
    return out


def timing_summary(per_seq_stats: List[Dict]) -> Dict:
    pf = [x for s in per_seq_stats for x in s["per_frame"]]
    if not pf:
        return {}
    ms = np.array([t for _, t in pf]) * 1000
    return {"tracker_fps": float(1000.0 / ms.mean()), "mean_ms": float(ms.mean()), "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)), "mean_dets": float(np.mean([n for n, _ in pf])), "frames": len(pf)}


def run_experiment(cfg: Dict, split: Optional[str] = None, detsets: Optional[List[DetSet]] = None,
                   model=None, write: bool = True) -> Dict:
    """Track every sequence of `split` with cfg's method, evaluate, and persist."""
    set_seed(cfg["seed"])
    split = split or cfg["data"]["eval_split"]
    detsets = detsets if detsets is not None else get_detsets(cfg, split)
    if cfg["method"] == "rca" and model is None:
        model = ensure_model(cfg)
    rdir = ensure_dir(run_dir(cfg)) if write else None
    per_seq, stats_all, preds = {}, [], {}
    for ds in detsets:
        rows, stats = run_tracker(ds, cfg, model)
        preds[ds.spec.name] = rows
        stats_all.append(stats)
        if write:
            write_mot_txt(rdir / f"{ds.spec.name}.txt", rows)
        per_seq[ds.spec.name] = evaluate(gt_eval_rows(ds.spec), rows, (ds.spec.frame_start, ds.spec.frame_end), cfg["eval"]["iou_thr"])
        r = per_seq[ds.spec.name]
        log.info(f"[{cfg['name']}] {ds.spec.name}: HOTA {r['HOTA']:.1f} AssA {r['AssA']:.1f} IDF1 {r['IDF1']:.1f} "
                 f"MOTA {r['MOTA']:.1f} IDSW {r['IDSW']}")
    comb = combine(list(per_seq.values()))
    result = {"name": cfg["name"], "method": cfg["method"], "dataset": cfg["data"]["dataset"], "split": split,
              "det_source": cfg["data"]["det_source"], "embedder": cfg["data"]["embedder"], "backend": "own",
              "combined": public(comb), "per_sequence": {k: public(v) for k, v in per_seq.items()},
              "timing": timing_summary(stats_all), "config": cfg}
    if write:
        save_json(result, rdir / "metrics.json")
    result["_preds"] = preds
    return result
