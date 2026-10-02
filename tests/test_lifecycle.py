import numpy as np

from rcamot.algorithm.tracker import MOTTracker
from rcamot.algorithm.track import TrackState
from rcamot.utils import load_config


def cfg(method="b0", **over):
    return load_config("configs/default.yaml", dataset="synthetic", overrides={"method": method, **over})


def box(x):
    return np.array([[x, 100, x + 40, 200.0]])


def test_birth_confirmation_and_continuity():
    tr = MOTTracker(cfg())
    ids = []
    for t in range(1, 8):
        out = tr.update(t, box(10 + 3 * t), np.array([0.9]), None)
        ids.append([o[0] for o in out])
    assert ids[0] == [1] and all(i == [1] for i in ids[1:])


def test_new_track_needs_second_hit_after_first_frame():
    tr = MOTTracker(cfg())
    tr.update(1, box(10), np.array([0.9]), None)
    out2 = tr.update(2, np.vstack([box(13), box(300)]), np.array([0.9, 0.9]), None)
    assert sorted(o[0] for o in out2) == [1]                       # the newcomer is still tentative
    out3 = tr.update(3, np.vstack([box(16), box(300)]), np.array([0.9, 0.9]), None)
    assert sorted(o[0] for o in out3) == [1, 2]


def test_lost_then_removed_after_max_age():
    tr = MOTTracker(cfg(tracker={"max_age": 5}))
    for t in range(1, 4):
        tr.update(t, box(10 + 3 * t), np.array([0.9]), None)
    for t in range(4, 10):
        tr.update(t, np.zeros((0, 4)), np.zeros(0), None)
    assert len(tr.tracks) == 0


def test_lost_track_is_reassociated_with_same_id():
    tr = MOTTracker(cfg())
    for t in range(1, 6):
        tr.update(t, box(10 + 3 * t), np.array([0.9]), None)
    for t in range(6, 11):
        out = tr.update(t, np.zeros((0, 4)), np.zeros(0), None)
        assert out == []
    states = {x.state for x in tr.tracks}
    assert states == {TrackState.LOST}
    out = tr.update(11, box(10 + 3 * 11), np.array([0.9]), None)
    assert [o[0] for o in out] == [1]


def test_low_score_detection_keeps_track_alive_stage2():
    tr = MOTTracker(cfg())
    for t in range(1, 4):
        tr.update(t, box(10 + 3 * t), np.array([0.9]), None)
    out = tr.update(4, box(22), np.array([0.3]), None)           # below high_thresh, above low_thresh
    assert [o[0] for o in out] == [1]


def test_low_score_never_spawns_tracks():
    tr = MOTTracker(cfg())
    tr.update(1, box(10), np.array([0.3]), None)
    assert tr.tracks == []


def test_unknown_method_and_missing_model_raise():
    import pytest
    with pytest.raises(ValueError):
        MOTTracker(cfg("b0") | {"method": "nope"})
    with pytest.raises(ValueError, match="fitted RCAModel"):
        MOTTracker(cfg("rca"))
