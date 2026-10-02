import numpy as np

from rcamot.algorithm.geometry import (center_dist_matrix, ioa_matrix, iou_matrix, max_other_ioa, xyah_to_xyxy, xyxy_to_xyah)


def test_iou_basic():
    a = np.array([[0, 0, 10, 10.0]])
    b = np.array([[0, 0, 10, 10.0], [5, 0, 15, 10.0], [20, 20, 30, 30.0]])
    m = iou_matrix(a, b)
    assert np.allclose(m[0], [1.0, 50 / 150, 0.0])


def test_iou_empty():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


def test_ioa_is_directional():
    small, big = np.array([[2, 2, 4, 4.0]]), np.array([[0, 0, 10, 10.0]])
    assert np.isclose(ioa_matrix(small, big)[0, 0], 1.0)       # small is fully covered by big
    assert np.isclose(ioa_matrix(big, small)[0, 0], 4 / 100)   # big is barely covered by small


def test_max_other_ioa_excludes_self():
    b = np.array([[0, 0, 10, 10.0], [5, 0, 15, 10.0], [100, 100, 110, 110.0]])
    o = max_other_ioa(b)
    assert np.allclose(o, [0.5, 0.5, 0.0])
    assert max_other_ioa(b[:1]).tolist() == [0.0]


def test_xyah_roundtrip():
    b = np.array([[10, 20, 50, 120.0]])
    assert np.allclose(xyah_to_xyxy(xyxy_to_xyah(b)), b)


def test_center_dist_scale_invariant():
    a, b = np.array([[0, 0, 10, 10.0]]), np.array([[10, 0, 20, 10.0]])
    d1 = center_dist_matrix(a, b)[0, 0]
    d2 = center_dist_matrix(a * 3, b * 3)[0, 0]
    assert np.isclose(d1, d2) and d1 > 0
