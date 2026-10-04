"""Reliability-Calibrated Association (RCA) model.

    M_ij   = b_iou*iou + b_ctr*ctr + b_tsu*tsu + b_iou_tsu*iou*tsu         (motion evidence)
    lam_ij = softplus(g0 + g_occ*occ + g_margin*margin + g_score*score)    (reliability-conditioned appearance weight)
    z_ij   = b0 + M_ij + lam_ij * (d0 - d_ij)                              (log-odds of "same identity")
    P_ij   = sigmoid(a * z_ij + b)                                         (Platt-calibrated posterior)

~10 parameters, fitted by weighted logistic regression (L-BFGS, analytic gradients), CPU only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from ..utils import resolve_path

PARAM_NAMES = ["b0", "b_iou", "b_ctr", "b_tsu", "b_iou_tsu", "g0", "g_occ", "g_margin", "g_score", "d0"]
INIT = np.array([-3.0, 4.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.3])
D0_BOUNDS = (0.0, 1.0)


def softplus(x):
    return np.logaddexp(0.0, x)


def free_mask(use_occ: bool = True, use_margin: bool = True, use_score: bool = True) -> np.ndarray:
    m = np.ones(len(PARAM_NAMES), dtype=bool)
    m[PARAM_NAMES.index("g_occ")] = use_occ
    m[PARAM_NAMES.index("g_margin")] = use_margin
    m[PARAM_NAMES.index("g_score")] = use_score
    return m


@dataclass
class RCAModel:
    params: np.ndarray = field(default_factory=lambda: INIT.copy())
    platt_a: float = 1.0
    platt_b: float = 0.0
    switches: Dict[str, bool] = field(default_factory=lambda: {"use_occ": True, "use_margin": True, "use_score": True})
    meta: Dict = field(default_factory=dict)
    d0_mode: str = "free_bounded"
    tau_bins: Dict[str, float] = field(default_factory=dict)
    platt_bins: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # ----- forward -----
    def _p(self, name: str) -> float:
        return float(self.params[PARAM_NAMES.index(name)])

    def lam(self, f: Dict[str, np.ndarray]) -> np.ndarray:
        g = (self._p("g0") + self._p("g_occ") * f["occ"] + self._p("g_margin") * f["margin"]
             + self._p("g_score") * f["score"])
        return softplus(g)

    def logit(self, f: Dict[str, np.ndarray]) -> np.ndarray:
        motion = (self._p("b0") + self._p("b_iou") * f["iou"] + self._p("b_ctr") * f["ctr"]
                  + self._p("b_tsu") * f["tsu"] + self._p("b_iou_tsu") * f["iou"] * f["tsu"])
        return motion + self.lam(f) * (self._p("d0") - f["d"])

    def posterior(self, f: Dict[str, np.ndarray], calibrated: bool = True) -> np.ndarray:
        z = self.logit(f)
        p = expit(self.platt_a * z + self.platt_b) if calibrated else expit(z)
        if calibrated and self.platt_bins:
            p = np.asarray(p).copy()
            frames = np.exp(np.asarray(f["tsu"]))
            for key, (a, b) in self.platt_bins.items():
                mask = gap_bin_mask(frames, key)
                p[mask] = expit(a * z[mask] + b)
        return p

    def threshold(self, f: Dict[str, np.ndarray], default: float) -> np.ndarray:
        frames = np.exp(np.asarray(f["tsu"]))
        out = np.full(frames.shape, float(default), dtype=np.float64)
        for key, tau in self.tau_bins.items():
            out[gap_bin_mask(frames, key)] = float(tau)
        return out

    # ----- persistence -----
    def to_dict(self) -> Dict:
        return {"param_names": PARAM_NAMES, "params": [float(x) for x in self.params],
                "platt_a": self.platt_a, "platt_b": self.platt_b, "switches": self.switches, "meta": self.meta,
                "d0_mode": self.d0_mode, "tau_bins": self.tau_bins,
                "platt_bins": {k: list(v) for k, v in self.platt_bins.items()}}

    def save(self, path) -> Path:
        p = resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p

    @staticmethod
    def load(path) -> "RCAModel":
        p = resolve_path(path)
        if not p.exists():
            raise FileNotFoundError(f"RCA parameter file not found: {p}. Run `python main.py train --config configs/rca.yaml` first.")
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return RCAModel(np.array(d["params"], dtype=np.float64), d.get("platt_a", 1.0), d.get("platt_b", 0.0),
                        d.get("switches", {}), d.get("meta", {}), d.get("d0_mode", "free_bounded"),
                        d.get("tau_bins", {}), {k: tuple(v) for k, v in d.get("platt_bins", {}).items()})


GAP_BINS = (("1", 1, 1), ("2-5", 2, 5), ("6-15", 6, 15), ("16+", 16, np.inf))


def gap_bin_mask(frames, key: str):
    # `tsu` is stored as log(frame_count); round away exp(log(n)) floating-point drift at cut points.
    values = np.rint(np.asarray(frames)).astype(np.int64)
    for name, lo, hi in GAP_BINS:
        if name == key:
            return (values >= lo) & (values <= hi)
    return np.zeros(values.shape, dtype=bool)


def fit_gap_thresholds(posteriors, labels, frames, alpha: float, fallback: float,
                       min_each_class: int = 20) -> Dict[str, float]:
    """Per-gap negative-posterior quantiles; sparse bins use the configured global threshold."""
    p, y, fr = np.asarray(posteriors), np.asarray(labels).astype(bool), np.asarray(frames)
    result = {}
    for key, _, _ in GAP_BINS:
        mask = gap_bin_mask(fr, key)
        pos, neg = mask & y, mask & ~y
        result[key] = (float(np.quantile(p[neg], 1.0 - alpha, method="higher"))
                       if pos.sum() >= min_each_class and neg.sum() >= min_each_class else float(fallback))
    return result


# ----- fitting -----
def fit_rca(X: Dict[str, np.ndarray], y: np.ndarray, free: Optional[np.ndarray] = None, l2: float = 1e-3,
            init: Optional[np.ndarray] = None, max_iter: int = 300,
            fixed: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, Dict]:
    """Weighted-free logistic regression with analytic gradients (L-BFGS-B)."""
    free = free_mask() if free is None else np.asarray(free, dtype=bool)
    theta0 = (INIT if init is None else np.asarray(init)).copy()
    fixed = dict(fixed or {})
    for name, value in fixed.items():
        if name not in PARAM_NAMES:
            raise ValueError(f"Unknown fixed RCA parameter: {name}")
        theta0[PARAM_NAMES.index(name)] = float(value)
    theta0[~free] = 0.0  # masked gamma parameters are pinned to zero
    optimize = free.copy()
    optimize[[PARAM_NAMES.index(k) for k in fixed]] = False
    y = np.asarray(y, dtype=np.float64)
    n = max(len(y), 1)
    iou, ctr, tsu, d, occ, mar, sc = (X[k] for k in ("iou", "ctr", "tsu", "d", "occ", "margin", "score"))
    idx = np.where(optimize)[0]

    def unpack(t):
        full = theta0.copy()
        full[idx] = t
        return full

    def fun(t):
        th = unpack(t)
        b0, b1, b2, b3, b4, g0, go, gm, gs, d0 = th
        g = g0 + go * occ + gm * mar + gs * sc
        lam = softplus(g)
        z = b0 + b1 * iou + b2 * ctr + b3 * tsu + b4 * iou * tsu + lam * (d0 - d)
        loss = np.sum(np.logaddexp(0.0, z) - y * z) / n + 0.5 * l2 * np.sum(th[1:] ** 2)
        dz = (expit(z) - y) / n
        dg = dz * (d0 - d) * expit(g)
        grad = np.array([dz.sum(), (dz * iou).sum(), (dz * ctr).sum(), (dz * tsu).sum(), (dz * iou * tsu).sum(),
                         dg.sum(), (dg * occ).sum(), (dg * mar).sum(), (dg * sc).sum(), (dz * lam).sum()])
        grad[1:] += l2 * th[1:]
        return loss, grad[idx]

    # d0 (the appearance-distance "neutral point") is bounded so that lam*d0 cannot act as a free
    # margin/score-dependent intercept: reliability features must modulate the appearance TERM, not add evidence.
    bounds = [(D0_BOUNDS if PARAM_NAMES[i] == "d0" else (None, None)) for i in idx]
    res = minimize(fun, theta0[idx], jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": max_iter})
    th = unpack(res.x)
    return th, {"loss": float(res.fun), "iterations": int(res.nit), "converged": bool(res.success), "n": int(len(y))}


def fit_platt(z: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Platt scaling: P = sigmoid(a*z + b) fitted by maximum likelihood on a held-out calibration set."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def fun(t):
        a, b = t
        s = a * z + b
        loss = -np.mean(y * log_expit(s) + (1 - y) * log_expit(-s))
        p = expit(s)
        return loss, np.array([np.mean((p - y) * z), np.mean(p - y)])

    res = minimize(fun, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


# ----- calibration metrics -----
def calibration_metrics(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> Dict:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece, bins = 0.0, []
    for b in range(n_bins):
        sel = ids == b
        if sel.sum() == 0:
            continue
        conf, acc = p[sel].mean(), y[sel].mean()
        ece += sel.mean() * abs(conf - acc)
        bins.append({"bin": b, "count": int(sel.sum()), "confidence": float(conf), "accuracy": float(acc)})
    return {"ECE": float(ece), "Brier": float(np.mean((p - y) ** 2)),
            "NLL": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "n": int(len(y)),
            "pos_rate": float(y.mean()) if len(y) else float("nan"), "bins": bins}
