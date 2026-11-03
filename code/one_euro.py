"""One Euro filter (Casiez et al., CHI 2012) for one signal with timestamps.

Used to smooth joint-angle signals from MediaPipe pose in evaluate2.py.
"""
import math

EURO_MIN_CUTOFF = 0.8   # Hz, smoothing when still (lower = steadier)
EURO_BETA = 0.02        # how quickly smoothing opens up with speed (deg/s)


class OneEuro:
    def __init__(self, min_cutoff=EURO_MIN_CUTOFF, beta=EURO_BETA, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        dx = (x - self.x) / dt
        self.dx += self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        return self.x
