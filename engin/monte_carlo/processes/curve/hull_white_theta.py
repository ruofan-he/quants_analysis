import numpy as np
from scipy.interpolate import CubicSpline


class HullWhiteTheta:
    """Fit the time-dependent Hull-White drift to an initial discount curve."""

    def __init__(self, times, dfs, alpha, sigma):
        times = np.asarray(times, dtype=float)
        dfs = np.asarray(dfs, dtype=float)

        if times.ndim != 1 or dfs.ndim != 1 or len(times) != len(dfs):
            raise ValueError("times and dfs must be one-dimensional arrays of equal length")
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("times must contain at least two strictly increasing values")
        if np.any(dfs <= 0):
            raise ValueError("discount factors must be positive")

        self.times = times
        self.log_dfs = np.log(dfs)
        self.log_df_spline = CubicSpline(times, self.log_dfs, extrapolate=True)
        self.alpha = alpha if callable(alpha) else lambda t: float(alpha)
        self.sigma = sigma if callable(sigma) else lambda t: float(sigma)

    def fwd_rate(self, t):
        return -self.log_df_spline.derivative(1)(t)

    def dfwd_dt(self, t):
        return -self.log_df_spline.derivative(2)(t)

    def theta(self, t):
        a = self.alpha(t)
        s = self.sigma(t)
        if np.isclose(a, 0.0):
            raise ValueError("This theta implementation assumes non-zero mean reversion alpha")
        f = self.fwd_rate(t)
        dfdt = self.dfwd_dt(t)
        dvar_dt = 2.0 * s**2 / a * (1.0 - np.exp(-a*t)) * np.exp(-a*t)
        return dfdt + a*f + 0.5*dvar_dt

    def df(self, t):
        return np.exp(self.log_df_spline(t))
