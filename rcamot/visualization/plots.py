"""Matplotlib figures (Agg backend; no display needed)."""
from __future__ import annotations

from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils import resolve_path

COLORS = {"b0": "#8c8c8c", "b1": "#1f77b4", "b2": "#2ca02c", "b3": "#ff7f0e", "rca": "#d62728"}


def _save(fig, path):
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_reliability(bins_by_variant: Dict[str, List[Dict]], path, title="Reliability diagram (held-out pairs)"):
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, bins in bins_by_variant.items():
        if bins:
            ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", ms=4, label=name)
    ax.set_xlabel("predicted match probability")
    ax.set_ylabel("empirical match frequency")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_s1(agg: pd.DataFrame, path, metric: str = "IDF1"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, g in agg.groupby("method"):
        g = g.sort_values("kappa")
        ax.plot(g["kappa"], g[metric], "o-", label=m, color=COLORS.get(m))
    ax.set_xlabel("crowding kappa (mean #GT pairs with IoU>0.4 / frame), bin means")
    ax.set_ylabel(metric)
    ax.set_title(f"S1: {metric} vs crowding")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_s2(df: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, col, ttl in zip(axes, ("reassoc_rate", "wrong_id_rate"), ("re-association success", "wrong-ID rate (identity drift)")):
        for m, g in df.groupby("method"):
            g = g.sort_values("gap_frames")
            ax.plot(g["gap_frames"], g[col], "o-", label=m, color=COLORS.get(m))
        ax.set_xlabel("injected disappearance (frames)")
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
    axes[0].legend()
    return _save(fig, path)


def plot_ablation(df: pd.DataFrame, path, metrics=("AssA", "IDF1", "IDSW")):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        ax.barh(df["variant"], df[m], color="#d62728")
        ax.set_title(m)
        ax.invert_yaxis()
    return _save(fig, path)


def plot_runtime(df: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, g in df.groupby("method"):
        ax.plot(g["dets_bin"].astype(str), g["mean_ms"], "o-", label=m, color=COLORS.get(m))
    ax.set_xlabel("detections per frame")
    ax.set_ylabel("tracker ms / frame (excl. detector & embedding)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_s2_curve(points: pd.DataFrame, curve: pd.DataFrame, path):
    """Operating curve: wrong-ID rate (x, 'false accept') vs re-association rate (y, 'true accept') per gap length.
    Baselines are single points; RCA is a curve obtained by sweeping the accept threshold."""
    ks = sorted(points["gap_frames"].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(3.6 * len(ks), 3.8), squeeze=False)
    for ax, k in zip(axes[0], ks):
        c = curve[curve.gap_frames == k].copy()
        c["tau"] = c["method"].str.extract(r"tau=([0-9.]+)").astype(float)
        c = c.sort_values("tau")
        ax.plot(c["wrong_id_rate"], c["reassoc_rate"], "o-", color=COLORS["rca"], label="RCA (tau sweep)")
        for _, r in c.iterrows():
            ax.annotate(f"{r['tau']:g}", (r["wrong_id_rate"], r["reassoc_rate"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
        for _, r in points[(points.gap_frames == k) & (points.method != "rca")].iterrows():
            ax.plot(r["wrong_id_rate"], r["reassoc_rate"], "s", color=COLORS.get(r["method"]), label=r["method"])
        ax.set_title(f"gap = {k} frames")
        ax.set_xlabel("wrong-ID rate (false accept)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("re-association rate (true accept)")
    axes[0][0].legend(fontsize=7)
    return _save(fig, path)
