"""Fit + calibrate the RCA model (seconds on CPU) and evaluate calibration on held-out pairs."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..inference.pipeline import get_detsets
from ..models.rca import PARAM_NAMES, RCAModel, calibration_metrics, fit_platt, fit_rca, free_mask
from ..utils import config_hash, get_logger, save_json, set_seed
from .mine_pairs import concat_pairs, mine_pairs

log = get_logger()


_PAIR_CACHE: Dict[str, Dict[str, np.ndarray]] = {}


def mined_pairs(cfg: Dict, split: str) -> Dict[str, np.ndarray]:
    """Mine (and memoise in-process) oracle-track pairs; ablations that only change the model reuse them."""
    key = config_hash({"d": cfg["data"], "s": split, "t": cfg["tracker"], "g": cfg["gate"], "m": cfg["rca"]["margin_clip"],
                       "b": cfg["rca"]["fit_block"], "seed": cfg["seed"]})
    if key not in _PAIR_CACHE:
        _PAIR_CACHE[key] = concat_pairs([mine_pairs(ds, cfg) for ds in get_detsets(cfg, split)])
    return _PAIR_CACHE[key]


def train_rca(cfg: Dict) -> Tuple[RCAModel, Dict]:
    set_seed(cfg["seed"])
    r = cfg["rca"]
    pairs = mined_pairs(cfg, "train")
    y = pairs["label"]
    fit_sel = (pairs["block"] % 2) == 0
    cal_sel = ~fit_sel
    if cal_sel.sum() < 200 or fit_sel.sum() < 200:
        log.warning("Too few pairs to split fit/calibration blocks; using all pairs for both (calibration will be optimistic).")
        fit_sel = cal_sel = np.ones(len(y), bool)
    free = free_mask(r["use_occ"], r["use_margin"], r["use_score"])
    sub = lambda sel: {k: pairs[k][sel] for k in ("iou", "ctr", "tsu", "d", "occ", "margin", "score")}
    theta, info = fit_rca(sub(fit_sel), y[fit_sel], free, r["l2"])
    model = RCAModel(theta, switches={"use_occ": r["use_occ"], "use_margin": r["use_margin"], "use_score": r["use_score"]})
    z_cal = model.logit(sub(cal_sel))
    model.platt_a, model.platt_b = fit_platt(z_cal, y[cal_sel])
    model.meta = {"n_pairs": int(len(y)), "pos_rate": float(y.mean()), "n_fit": int(fit_sel.sum()), "n_calib": int(cal_sel.sum()),
                  "fit": info, "dataset": cfg["data"]["dataset"], "embedder": cfg["data"]["embedder"],
                  "det_source": cfg["data"]["det_source"], "config_hash": config_hash({"t": cfg["tracker"], "g": cfg["gate"], "r": r})}
    path = model.save(r["model_path"])
    log.info(f"[train] {len(y)} pairs (pos rate {y.mean():.3f}); params = " +
             ", ".join(f"{n}={v:.3f}" for n, v in zip(PARAM_NAMES, theta)))
    log.info(f"[train] Platt a={model.platt_a:.3f} b={model.platt_b:.3f}; saved {path}")
    return model, {"info": info, "n_pairs": len(y)}


def evaluate_calibration(cfg: Dict, model: RCAModel, out_csv, fig_path=None) -> Dict:
    """ECE / Brier / NLL of raw vs Platt-calibrated posteriors on pairs mined from the held-out (val) split."""
    import pandas as pd
    from ..utils import ensure_dir, resolve_path
    pairs = mined_pairs(cfg, cfg["data"]["eval_split"])
    y = pairs["label"]
    feats = {k: pairs[k] for k in ("iou", "ctr", "tsu", "d", "occ", "margin", "score")}
    raw = calibration_metrics(model.posterior(feats, calibrated=False), y)
    cal = calibration_metrics(model.posterior(feats, calibrated=True), y)
    rows = [{"dataset": cfg["data"]["dataset"], "variant": v, "ECE": m["ECE"], "Brier": m["Brier"], "NLL": m["NLL"],
             "n_pairs": m["n"], "pos_rate": m["pos_rate"]} for v, m in (("uncalibrated", raw), ("platt", cal))]
    df = pd.DataFrame(rows)
    p = resolve_path(out_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    if fig_path is not None:
        from ..visualization.plots import plot_reliability
        plot_reliability({"uncalibrated": raw["bins"], "platt": cal["bins"]}, fig_path)
    return {"uncalibrated": raw, "platt": cal}
