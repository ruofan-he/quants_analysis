import numpy as np
from scipy.interpolate import CubicSpline


class InitialCurve:
    """Initial discount curve."""

    def __init__(self, times, dfs):
        times = np.asarray(times, dtype=float)
        dfs = np.asarray(dfs, dtype=float)

        if times.ndim != 1 or dfs.ndim != 1 or len(times) != len(dfs):
            raise ValueError("times and dfs must be one-dimensional arrays of equal length")
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("times must contain at least two strictly increasing values")
        if np.any(dfs <= 0):
            raise ValueError("discount factors must be positive")

        self.times = times
        self.dfs = dfs
        self.log_dfs = np.log(dfs)

        self.log_df_spline = CubicSpline(
            times,
            self.log_dfs,
            extrapolate=True
        )

    def df(self, t):
        return np.exp(self.log_df_spline(t))

    def fwd_rate(self, t):
        return -self.log_df_spline.derivative(1)(t)

    def dfwd_dt(self, t):
        return -self.log_df_spline.derivative(2)(t)