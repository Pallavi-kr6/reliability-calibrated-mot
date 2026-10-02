"""Constant-velocity Kalman filter over (cx, cy, aspect, height) + velocities (DeepSORT/ByteTrack style)."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class KalmanFilter:
    def __init__(self, std_pos: float = 1.0 / 20, std_vel: float = 1.0 / 160):
        self.std_pos, self.std_vel = std_pos, std_vel
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.H = np.eye(4, 8)

    def initiate(self, meas: np.ndarray):
        mean = np.r_[meas, np.zeros(4)]
        h = meas[3]
        std = [2 * self.std_pos * h, 2 * self.std_pos * h, 1e-2, 2 * self.std_pos * h,
               10 * self.std_vel * h, 10 * self.std_vel * h, 1e-5, 10 * self.std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray):
        h = mean[3]
        std_p = [self.std_pos * h, self.std_pos * h, 1e-2, self.std_pos * h]
        std_v = [self.std_vel * h, self.std_vel * h, 1e-5, self.std_vel * h]
        Q = np.diag(np.square(np.r_[std_p, std_v]))
        return self.F @ mean, self.F @ cov @ self.F.T + Q

    def project(self, mean: np.ndarray, cov: np.ndarray):
        h = mean[3]
        std = [self.std_pos * h, self.std_pos * h, 1e-1, self.std_pos * h]
        R = np.diag(np.square(std))
        return self.H @ mean, self.H @ cov @ self.H.T + R

    def update(self, mean: np.ndarray, cov: np.ndarray, meas: np.ndarray):
        pm, pc = self.project(mean, cov)
        K = np.linalg.solve(pc, (cov @ self.H.T).T).T  # Kalman gain (8x4)
        new_mean = mean + K @ (meas - pm)
        new_cov = cov - K @ pc @ K.T
        return new_mean, new_cov

    def gating_distance(self, mean: np.ndarray, cov: np.ndarray, meas: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance between the projected state and measurements (K,4)."""
        pm, pc = self.project(mean, cov)
        d = np.atleast_2d(meas) - pm
        c, low = cho_factor(pc, lower=True, check_finite=False)
        z = cho_solve((c, low), d.T, check_finite=False)
        return np.sum(d.T * z, axis=0)
