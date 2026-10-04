import numpy as np

from rcamot.models.rca import (PARAM_NAMES, RCAModel, calibration_metrics, fit_gap_thresholds, fit_platt, fit_rca, free_mask, gap_bin_mask, softplus)
from scipy.special import expit


def make(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    X = {k: rng.random(n) for k in ("iou", "ctr", "tsu", "d", "occ", "margin", "score")}
    X["margin"] = X["margin"] * 0.6 - 0.3
    lam = softplus(0.5 + 6.0 * X["margin"])  # appearance matters only when the margin is large
    z = -2 + 5 * X["iou"] + lam * (0.5 - X["d"])
    y = (rng.random(n) < expit(z)).astype(float)
    return X, y


def test_margin_modulated_lambda_improves_fit_and_is_positive():
    X, y = make(12000)
    th, info = fit_rca(X, y)
    _, info_off = fit_rca(X, y, free_mask(use_occ=True, use_margin=False, use_score=True))
    assert info["converged"]
    assert th[PARAM_NAMES.index("g_margin")] > 0.3
    assert info["loss"] < info_off["loss"] - 1e-4          # the reliability feature explains real structure
    assert th[PARAM_NAMES.index("b_iou")] > 2.0
    assert 0.0 <= th[PARAM_NAMES.index("d0")] <= 1.0


def test_masked_features_stay_zero():
    X, y = make(2000)
    th, _ = fit_rca(X, y, free_mask(use_occ=False, use_margin=False, use_score=False))
    assert th[PARAM_NAMES.index("g_occ")] == 0 and th[PARAM_NAMES.index("g_margin")] == 0 and th[PARAM_NAMES.index("g_score")] == 0


def test_platt_fixes_overconfidence():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 3, 20000)
    y = (rng.random(20000) < expit(0.4 * z)).astype(float)    # true slope 0.4, model claims slope 1
    a, b = fit_platt(z, y)
    assert 0.3 < a < 0.5 and abs(b) < 0.1
    assert calibration_metrics(expit(a * z + b), y)["ECE"] < calibration_metrics(expit(z), y)["ECE"]


def test_ece_of_calibrated_probabilities_is_small():
    rng = np.random.default_rng(2)
    p = rng.random(50000)
    y = (rng.random(50000) < p).astype(float)
    assert calibration_metrics(p, y)["ECE"] < 0.02


def test_model_save_load_roundtrip(tmp_path):
    m = RCAModel(platt_a=1.3, platt_b=-0.2)
    m.params[1] = 4.4
    p = m.save(tmp_path / "m.json")
    m2 = RCAModel.load(p)
    assert np.allclose(m.params, m2.params) and m2.platt_a == 1.3


def test_fixed_d0_is_not_optimized():
    X, y = make(1200, 4)
    value = 0.123
    theta, _ = fit_rca(X, y, fixed={"d0": value})
    assert theta[PARAM_NAMES.index("d0")] == value


def test_legacy_model_json_loads_without_new_fields(tmp_path):
    import json
    legacy = {"params": [float(x) for x in np.zeros(len(PARAM_NAMES))], "platt_a": 1.2, "platt_b": -0.1}
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(legacy))
    loaded = RCAModel.load(p)
    assert loaded.platt_a == 1.2 and loaded.tau_bins == {} and loaded.d0_mode == "free_bounded"


def test_gap_threshold_quantile_and_sparse_fallback():
    # The first bin has 20 negatives and 20 positives; the second is intentionally sparse.
    neg = np.linspace(0.1, 0.9, 20)
    p = np.r_[neg, np.full(20, 0.95), np.linspace(0.1, 1.0, 10)]
    y = np.r_[np.zeros(20), np.ones(20), np.r_[np.zeros(5), np.ones(5)]]
    frames = np.r_[np.ones(40), np.full(10, 2)]
    tau = fit_gap_thresholds(p, y, frames, alpha=0.1, fallback=0.37)
    assert tau["1"] == np.quantile(neg, 0.9, method="higher")
    assert tau["2-5"] == 0.37  # fewer than 20 examples in each class
    assert gap_bin_mask(np.array([np.exp(np.log(16.0))]), "16+")[0]


def test_per_gap_platt_uses_gap_specific_transform():
    m = RCAModel()
    m.platt_bins = {"1": (2.0, 0.0)}
    f = {"iou": np.zeros(2), "ctr": np.zeros(2), "tsu": np.array([0.0, np.log(2.0)]),
         "d": np.ones(2) * 0.5, "occ": np.zeros(2), "margin": np.zeros(2), "score": np.zeros(2)}
    p = m.posterior(f)
    assert p[0] != p[1]


def test_missing_model_gives_helpful_error(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError, match="main.py train"):
        RCAModel.load(tmp_path / "nope.json")
