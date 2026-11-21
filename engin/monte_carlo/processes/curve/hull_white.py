import numpy as np
from scipy.integrate import trapezoid
from .base import CurveProcess


class HullWhiteProcess(CurveProcess):
    """Euler-discretized Hull-White one-factor process."""

    def __init__(self, initial_value, alpha, sigma, theta):
        super().__init__(initial_value)
        self.alpha = alpha if callable(alpha) else lambda t: float(alpha)
        self.sigma = sigma if callable(sigma) else lambda t: float(sigma)
        self.theta = theta

    def step(self, t, x, dt, dW, context=None):
        a = self.alpha(t)
        s = self.sigma(t)
        return x + (self.theta(t) - a*x)*dt + s*dW

    def B(self, t, maturity):
        a = self.alpha(t)
        tau = np.asarray(maturity) - t
        if np.isclose(a, 0.0):
            return tau
        return -np.expm1(-a*tau) / a

    def A(self, t, maturity, n_int=101):
        """Log-A term for P(t,T)=exp(A(t,T)-B(t,T)*x_t)."""
        if t >= maturity:
            return 0.0

        u = np.linspace(t, maturity, n_int)
        b = self.B(u, maturity)
        theta_values = np.asarray([self.theta(float(x)) for x in u])
        sigma_values = np.asarray([self.sigma(float(x)) for x in u])

        integrand = -b*theta_values + 0.5*(sigma_values*b)**2
        return float(trapezoid(integrand, u))

    def zcb(self, t, x_t, maturity):
        if t > maturity:
            return np.zeros_like(x_t, dtype=float)
        if np.isclose(t, maturity):
            return np.ones_like(x_t, dtype=float)
        return np.exp(self.A(t, maturity) - self.B(t, maturity)*x_t)
