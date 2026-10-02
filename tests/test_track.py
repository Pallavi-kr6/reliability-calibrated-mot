import numpy as np

from rcamot.algorithm.kalman import KalmanFilter
from rcamot.algorithm.track import Track, TrackState


def mk(emb=None):
    return Track(1, np.array([0, 0, 20, 60.0]), 0.9, emb if emb is not None else np.array([1.0, 0, 0]), 1, KalmanFilter())


def test_ema_prototype_is_unit_and_moves_slowly():
    t = mk()
    t.predict(2)
    t.update(np.array([1, 0, 21, 60.0]), 0.9, np.array([0, 1.0, 0]), 2, alpha=0.9)
    assert np.isclose(np.linalg.norm(t.proto), 1.0)
    assert t.proto[0] > 0.9 and t.proto[1] > 0


def test_alpha_one_freezes_prototype():
    t = mk()
    before = t.proto.copy()
    t.predict(2)
    t.update(np.array([1, 0, 21, 60.0]), 0.9, np.array([0, 1.0, 0]), 2, alpha=1.0)
    assert np.allclose(t.proto, before)


def test_state_transitions_and_tsu():
    t = mk()
    assert t.state == TrackState.TENTATIVE
    t.predict(2)
    t.update(np.array([1, 0, 21, 60.0]), 0.9, None, 2, 0.9, min_hits=2)
    assert t.state == TrackState.TRACKED
    t.mark_lost()
    assert t.state == TrackState.LOST and t.tsu(6) == 4
    t.predict(3); t.update(np.array([1, 0, 21, 60.0]), 0.9, None, 3, 0.9)
    assert t.state == TrackState.TRACKED


def test_lost_velocity_is_damped():
    t = mk()
    for f in range(2, 8):
        t.predict(f); t.update(np.array([5.0 * f, 0, 20 + 5.0 * f, 60.0]), 0.9, None, f, 0.9)
    v0 = abs(t.mean[4])
    for f in range(8, 20):
        t.predict(f)
    assert abs(t.mean[4]) < 0.5 * v0
