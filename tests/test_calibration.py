import numpy as np

from rcamot.models.rca import (PARAM_NAMES, RCAModel, calibration_metrics, fit_platt, fit_rca, free_mask, softplus)
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


def test_missing_model_gives_helpful_error(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError, match="main.py train"):
        RCAModel.load(tmp_path / "nope.json")
