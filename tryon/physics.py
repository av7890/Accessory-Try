"""Damped pendulum for swinging earrings, driven by the horizontal acceleration of the pivot (the ear)."""
import math

import numpy as np


class Pendulum:
    """theta = angle from straight-down (rad), positive = bob towards +x.

    theta'' = -(g/L) sin(theta) - c*theta' - (a_x/L) cos(theta)
    a_x is the pivot's horizontal acceleration in px/s^2.
    """

    DAMPING = 3.5
    MAX_THETA = 1.0

    def __init__(self):
        self.theta = self.omega = 0.0
        self.t = self.px = self.pv = None

    def step(self, pivot_x: float, length: float, g: float, t: float) -> float:
        if self.t is None:
            self.t, self.px, self.pv = t, pivot_x, 0.0
            return 0.0
        dt = min(max(t - self.t, 1e-3), 0.05)
        v = (pivot_x - self.px) / dt
        a = float(np.clip((v - self.pv) / dt, -8 * g, 8 * g))
        length = max(length, 1.0)
        n = 4
        h = dt / n
        for _ in range(n):
            acc = -(g / length) * math.sin(self.theta) - self.DAMPING * self.omega - (a / length) * math.cos(self.theta)
            self.omega += acc * h
            self.theta += self.omega * h
        self.theta = float(np.clip(self.theta, -self.MAX_THETA, self.MAX_THETA))
        self.t, self.px, self.pv = t, pivot_x, v
        return self.theta
