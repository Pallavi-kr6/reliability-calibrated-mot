"""Paired bootstrap confidence intervals over windows / sequences."""
from __future__ import annotations

from typing import Dict

import numpy as np


MIN_UNITS = 10  # below this many paired windows/sequences a bootstrap CI is not trustworthy


def paired_bootstrap(diff: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """CI of the mean of per-unit paired differences (e.g. RCA - baseline per window). Resamples units.
    `excludes_zero` is only ever True when at least MIN_UNITS units exist (`reliable`)."""
    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "excludes_zero": False, "reliable": False}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi), "n": int(n),
            "excludes_zero": bool((lo > 0 or hi < 0) and n >= MIN_UNITS), "reliable": bool(n >= MIN_UNITS)}
