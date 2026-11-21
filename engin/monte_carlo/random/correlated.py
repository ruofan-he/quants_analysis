import numpy as np


class CorrelatedBrownian:
    """Generate correlated Brownian increments in a stable, named factor order."""

    def __init__(self, factor_names, correlation, seed=None):
        self.factor_names = tuple(factor_names)
        corr = np.asarray(correlation, dtype=float)

        n = len(self.factor_names)
        if corr.shape != (n, n):
            raise ValueError(f"correlation must have shape ({n}, {n})")
        if not np.allclose(corr, corr.T, atol=1e-12):
            raise ValueError("correlation matrix must be symmetric")
        if not np.allclose(np.diag(corr), 1.0, atol=1e-12):
            raise ValueError("correlation matrix must have unit diagonal")

        self.cholesky = np.linalg.cholesky(corr)
        self.rng = np.random.default_rng(seed)

    def generate(self, n_paths, dt):
        z = self.rng.standard_normal((n_paths, len(self.factor_names)))
        return z @ self.cholesky.T * np.sqrt(dt)
