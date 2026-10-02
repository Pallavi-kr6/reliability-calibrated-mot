import numpy as np

from rcamot.algorithm.features import candidate_margin


def test_margin_hand_computed():
    d = np.array([[0.10, 0.12], [0.50, 0.11]])
    m = candidate_margin(d, np.ones_like(d, bool))
    assert np.allclose(m, [[0.02, -0.02], [-0.3, 0.01]])  # (1,0) is clipped from -0.40


def test_unique_candidate_gets_max_margin():
    d = np.array([[0.2]])
    assert candidate_margin(d, np.ones((1, 1), bool))[0, 0] == 0.3


def test_gate_masks_competitors():
    d = np.array([[0.10, 0.05], [0.40, 0.45]])
    mask = np.array([[True, False], [True, True]])
    m = candidate_margin(d, mask)
    # pair (0,0): its only row competitor (0,1) is gated out; column competitor is track 1 with d=0.40 -> 0.30
    assert np.isclose(m[0, 0], 0.30)


def test_indistinguishable_candidates_have_near_zero_margin():
    d = np.full((3, 3), 0.3) + np.eye(3) * 0.001
    m = candidate_margin(d, np.ones((3, 3), bool))
    assert np.all(np.abs(m) < 0.01)


def test_empty_shapes():
    assert candidate_margin(np.zeros((0, 4)), np.zeros((0, 4), bool)).shape == (0, 4)
