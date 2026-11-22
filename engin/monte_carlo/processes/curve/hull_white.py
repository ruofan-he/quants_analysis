import numpy as np
from scipy.integrate import trapezoid
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from .base import CurveProcess



class HullWhiteTheta:
    """Fit the time-dependent Hull-White drift to an initial discount curve."""

    def __init__(
        self,
        initial_curve,
        alpha,
        sigma,
        max_time=50.0,
        grid_step=0.01,
    ):
        self.initial_curve = initial_curve
        self.alpha = alpha if callable(alpha) else lambda t: float(alpha)
        self.sigma = sigma if callable(sigma) else lambda t: float(sigma)

        # Precompute theta on a time grid
        self.times = np.arange(0.0, max_time + grid_step, grid_step)

        self.theta_values = np.array([self.theta_impl(t) for t in self.times])

        # Interpolation
        self._theta_spline = CubicSpline(
            self.times,
            self.theta_values,
            extrapolate=True,
        )

    def theta_impl(self, t):
        """Calculate theta(t) directly using numerical integration."""

        f = self.initial_curve.fwd_rate(t)
        dfdt = self.initial_curve.dfwd_dt(t)

        def integrand(u):
            alpha_integral, _ = quad(
                self.alpha, u, t
            )

            return (
                self.sigma(u) ** 2
                * np.exp(-2.0 * alpha_integral)
            )

        vol_adjustment, _ = quad(integrand, 0.0, t)

        return (
            dfdt
            + self.alpha(t) * f
            + vol_adjustment
        )

    def theta(self, t):
        """Return interpolated theta(t)."""
        return self._theta_spline(t)




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

        theta_t = self.theta(t)
        theta_next = self.theta(t + dt)

        # Predictor
        x_pred = x + (theta_t - a*x)*dt + s*dW

        # Corrector
        return x + (
            (theta_t - a*x)
            + (theta_next - a*x_pred)
    ) * 0.5 * dt + s*dW

    def discount_rate(self, t1, t2, x1, x2):
        """Rate used to accumulate a discount/survival factor."""

        return 0.5 * (x1 + x2)

    def B(self, t, maturity):
        def integrand(u):
            alpha_integral, _ = quad(
                self.alpha, t, u
            )
            return np.exp(-alpha_integral)

        return quad(integrand, t, maturity)[0]

    def A(self, t, maturity, n_int=201):
        """Log-A term for P(t,T)=exp(A(t,T)-B(t,T)*x_t)."""
        if t >= maturity:
            return 0.0

        u = np.linspace(t, maturity, n_int)
        b = np.asarray([self.B(float(ui), maturity) for ui in u])
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
