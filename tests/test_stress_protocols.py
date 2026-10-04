import numpy as np
import pandas as pd

from rcamot.evaluation import stress


def test_s2_uses_configured_max_age(monkeypatch):
    class Spec:
        frame_start, frame_end = 1, 5
        name = "dummy"

    class DS:
        spec = Spec()
        frames = np.array([], dtype=int)
        gt_id = np.array([], dtype=int)

        def drop(self, mask):
            return self

    seen = {}
    monkeypatch.setattr(stress, "pick_gaps", lambda *a, **kw: [])
    monkeypatch.setattr(stress, "run_tracker", lambda ds, cfg, model: (seen.update(max_age=cfg["tracker"]["max_age"]) or (np.zeros((0, 7)), {})))
    monkeypatch.setattr(stress, "gt_eval_rows", lambda spec: np.zeros((0, 9)))
    cfg = {"tracker": {"max_age": 30}, "stress": {"s2_max_age": 90}}
    stress.gap_injection(cfg, DS(), None, 60, 3, 0)
    assert seen["max_age"] == 90 and cfg["tracker"]["max_age"] == 30


def test_s1_window_auto_shrinks(monkeypatch):
    class Spec:
        frame_start, frame_end = 1, 2400

    class DS:
        spec = Spec()

    monkeypatch.setattr(stress, "gt_eval_rows", lambda spec: np.zeros((0, 9)))
    monkeypatch.setattr(stress, "window_stats", lambda *a: {"density": 1.0})
    window, reliable = stress.choose_s1_window([DS()], 100, 4, 10)
    assert window == 50 and reliable
    window, reliable = stress.choose_s1_window([DS()], 100, 10, 10)
    assert window == 40 and not reliable


def test_matched_far_interpolation_and_out_of_range_nan():
    baseline = pd.DataFrame([
        {"method": "b1", "gap_frames": 15, "wrong_id_rate": 0.05, "reassoc_rate": 0.6, "n_gaps": 100},
        {"method": "b2", "gap_frames": 15, "wrong_id_rate": 0.2, "reassoc_rate": 0.7, "n_gaps": 100},
    ])
    curve = pd.DataFrame([
        {"gap_frames": 15, "wrong_id_rate": 0.0, "reassoc_rate": 0.2},
        {"gap_frames": 15, "wrong_id_rate": 0.1, "reassoc_rate": 0.6},
    ])
    result = stress.matched_far_table(baseline, curve).set_index("baseline")
    assert np.isclose(result.loc["b1", "rca_reassoc_at_matched_far"], 0.4)
    assert np.isnan(result.loc["b2", "delta_reassoc"])


def test_gap_gate_operating_curve_shifts_all_bins_equally():
    shifted = stress.offset_tau_bins({"1": 0.07, "16+": 0.02}, -0.05)
    assert np.isclose(shifted["1"], 0.02) and shifted["16+"] == 0.0
