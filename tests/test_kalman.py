import numpy as np

from rcamot.algorithm.kalman import KalmanFilter


def test_constant_velocity_tracking():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([100.0, 100.0, 0.4, 100.0]))
    for t in range(1, 30):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, np.array([100.0 + 3 * t, 100.0, 0.4, 100.0]))
    assert abs(mean[4] - 3.0) < 0.3                      # learned the x velocity
    mean, cov = kf.predict(mean, cov)
    assert abs(mean[0] - (100 + 3 * 30)) < 2.0           # extrapolates one step ahead


def test_update_reduces_uncertainty_and_predict_grows_it():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([50.0, 50.0, 0.5, 80.0]))
    m2, c2 = kf.predict(mean, cov)
    assert np.trace(c2) > np.trace(cov)
    m3, c3 = kf.update(m2, c2, np.array([50.0, 50.0, 0.5, 80.0]))
    assert np.trace(c3) < np.trace(c2)


def test_gating_distance_zero_at_prediction_and_grows():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([50.0, 50.0, 0.5, 80.0]))
    d = kf.gating_distance(mean, cov, np.array([[50.0, 50.0, 0.5, 80.0], [90.0, 50.0, 0.5, 80.0]]))
    assert d[0] < 1e-9 and d[1] > d[0]
