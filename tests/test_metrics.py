import numpy as np
import pytest

from rcamot.evaluation.bootstrap import paired_bootstrap
from rcamot.evaluation.metrics import combine, evaluate


def grid_gt(n_frames=50, n_ids=5):
    return np.array([[t, i + 1, 10 + i * 60 + t * 1.5, 50 + i * 5, 30, 80, 1, 1, 1.0]
                     for t in range(1, n_frames + 1) for i in range(n_ids)])


def test_perfect_tracking_scores_100():
    gt = grid_gt()
    r = evaluate(gt, gt[:, :7], (1, 50))
    assert r["HOTA"] == pytest.approx(100) and r["IDF1"] == pytest.approx(100) and r["MOTA"] == pytest.approx(100) and r["IDSW"] == 0


def test_identity_swap_counts_two_switches():
    gt = grid_gt()
    p = gt[:, :7].copy()
    a, b = (p[:, 0] > 25) & (p[:, 1] == 1), (p[:, 0] > 25) & (p[:, 1] == 2)
    p[a, 1], p[b, 1] = 2, 1
    r = evaluate(gt, p, (1, 50))
    assert r["IDSW"] == 2 and r["IDF1"] == pytest.approx(80.0) and r["DetA"] == pytest.approx(100)


def test_fragmentation_counts_one_switch_and_lowers_assa_not_deta():
    gt = grid_gt()
    p = gt[:, :7].copy()
    p[(p[:, 0] > 25) & (p[:, 1] == 3), 1] = 99
    r = evaluate(gt, p, (1, 50))
    assert r["IDSW"] == 1 and r["AssA"] < 100 and r["DetA"] == pytest.approx(100)


def test_missing_and_false_positive_detections():
    gt = grid_gt()
    p = gt[gt[:, 1] != 5][:, :7]                                     # one identity never tracked
    r = evaluate(gt, p, (1, 50))
    assert r["FN"] == 50 and r["FP"] == 0 and r["MOTA"] == pytest.approx(80.0)
    p2 = np.vstack([gt[:, :7], [[10, 500, 400, 400, 30, 80, 0.9]]])  # hallucinated box
    assert evaluate(gt, p2, (1, 50))["FP"] == 1


def test_distractor_matches_are_not_penalised():
    gt = grid_gt(10, 2)
    dis = np.array([[t, 50, 300, 300, 30, 80, 1, 7, 1.0] for t in range(1, 11)])   # static-person distractor
    p = np.vstack([gt[:, :7], [[t, 77, 300, 300, 30, 80, 0.9] for t in range(1, 11)]])
    r = evaluate(np.vstack([gt, dis]), p, (1, 10))
    assert r["FP"] == 0


def test_combine_matches_joint_counts():
    g1, g2 = grid_gt(20, 3), grid_gt(30, 4)
    r1, r2 = evaluate(g1, g1[:, :7], (1, 20)), evaluate(g2, g2[:, :7], (1, 30))
    c = combine([r1, r2])
    assert c["num_gt"] == r1["num_gt"] + r2["num_gt"] and c["HOTA"] == pytest.approx(100)


def test_matches_official_trackeval(tmp_path):
    """Cross-check HOTA/IDF1/MOTA/IDSW against the bundled official TrackEval on a perturbed tracker output."""
    from rcamot.evaluation.trackeval_backend import evaluate_with_trackeval, find_trackeval
    from rcamot.datasets.registry import SequenceSpec
    try:
        find_trackeval()
    except FileNotFoundError:
        pytest.skip("TrackEval not available")
    gt = grid_gt(60, 6)
    d = tmp_path / "seq"
    (d / "gt").mkdir(parents=True)
    np.savetxt(d / "gt" / "gt.txt", gt, delimiter=",", fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%.2f")
    rng = np.random.default_rng(0)
    p = gt[:, :7].copy()
    p[:, 2:4] += rng.normal(0, 2, (len(p), 2))
    p = p[rng.random(len(p)) > 0.1]
    p[(p[:, 0] > 30) & (p[:, 1] == 2), 1] = 42
    a, b = (p[:, 0] > 45) & (p[:, 1] == 3), (p[:, 0] > 45) & (p[:, 1] == 4)
    p[a, 1], p[b, 1] = 4, 3
    spec = SequenceSpec("synthetic", "seq", d, 1, 60, 640, 360, 25, d / "img1", ".jpg")
    te = evaluate_with_trackeval([spec], {"seq": p}, {})["combined"]
    mine = evaluate(gt, p, (1, 60))
    for k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA"):
        assert te[k] == pytest.approx(mine[k], abs=0.05), k
    assert te["IDSW"] == mine["IDSW"]


def test_bootstrap_ci():
    r = paired_bootstrap(np.full(30, 2.0) + np.random.default_rng(0).normal(0, 0.1, 30), 500)
    assert r["lo"] > 0 and r["excludes_zero"] and r["reliable"]
    small = paired_bootstrap(np.array([2.0, 2.1, 1.9]), 200)
    assert small["lo"] > 0 and not small["excludes_zero"] and not small["reliable"]   # n<10: never "significant"
    z = paired_bootstrap(np.random.default_rng(1).normal(0, 1, 40), 500)
    assert z["lo"] < 0 < z["hi"]
