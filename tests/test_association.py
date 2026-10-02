import numpy as np

from rcamot.algorithm.association import BIG, assign, fused_cost, iou_cost, rca_cost
from rcamot.algorithm.features import PairFeatures


def pf(iou, d, occ=0.0, score=0.9, mask=None):
    iou, d = np.atleast_2d(iou), np.atleast_2d(d)
    z = np.zeros_like(iou)
    return PairFeatures(iou, z + 0.1, z, d, z + occ, z + 0.1, z + score, np.ones_like(iou, bool) if mask is None else mask)


CFG = {"b0_min_iou": 0.2, "fixed_lambda": 0.5, "fused_cost_thresh": 0.7, "app_gate": 0.5, "ioa_gate": 0.5}


def test_assign_respects_forbidden_pairs():
    cost = np.array([[0.1, 0.9], [0.2, 0.3]])
    valid = np.array([[True, False], [True, True]])
    matches, ut, ud = assign(cost, valid)
    assert sorted(matches) == [(0, 0), (1, 1)] and ut == [] and ud == []


def test_assign_unmatched_when_all_invalid():
    matches, ut, ud = assign(np.ones((2, 2)), np.zeros((2, 2), bool))
    assert matches == [] and ut == [0, 1] and ud == [0, 1]


def test_assign_empty():
    assert assign(np.zeros((0, 3)), np.zeros((0, 3), bool)) == ([], [], [0, 1, 2])


def test_b3_switches_appearance_off_when_overlapped():
    f = pf([[0.5]], [[0.9]], occ=0.8)                    # appearance is terrible but pair is heavily overlapped
    cost, valid, info = fused_cost(f, "b3", CFG, 0.5)
    assert info["lam"][0, 0] == 0.0 and valid[0, 0] and np.isclose(cost[0, 0], 0.5)
    f2 = pf([[0.5]], [[0.9]], occ=0.0)                   # not overlapped: bad appearance is rejected
    assert not fused_cost(f2, "b3", CFG, 0.5)[1][0, 0]


def test_b2_scales_lambda_with_confidence():
    lo = fused_cost(pf([[0.5]], [[0.1]], score=0.55), "b2", CFG, 0.5)[2]["lam"][0, 0]
    hi = fused_cost(pf([[0.5]], [[0.1]], score=0.95), "b2", CFG, 0.5)[2]["lam"][0, 0]
    assert lo < hi <= 0.5


def test_b1_prefers_matching_appearance():
    f = pf([[0.5, 0.5]], [[0.05, 0.4]])
    cost, valid, _ = fused_cost(f, "b1", CFG, 0.5)
    assert cost[0, 0] < cost[0, 1]


def test_rca_gate_and_cost_monotone():
    class M:
        def posterior(self, feats, calibrated=True):
            return np.array([[0.9, 0.3]])

        def lam(self, feats):
            return np.ones((1, 2))

    f = pf([[0.5, 0.5]], [[0.1, 0.1]])
    cost, valid, info = rca_cost(f, M(), {"accept_thresh": 0.5, "calibrate": True})
    assert valid.tolist() == [[True, False]] and cost[0, 0] < cost[0, 1]
    cost, valid, _ = rca_cost(f, M(), {"accept_thresh": 0.0, "calibrate": True})
    assert valid.all()


def test_iou_cost_threshold():
    cost, valid = iou_cost(np.array([[0.1, 0.4]]), 0.2)
    assert valid.tolist() == [[False, True]] and np.isclose(cost[0, 1], 0.6)
