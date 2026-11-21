import numpy as np
from .base import SpotProcess


class BlackScholesProcess(SpotProcess):
    """Risk-neutral GBM. The short-rate path is supplied by the engine context."""

    def __init__(self, initial_value, volatility, rate_key="rate"):
        super().__init__(initial_value)
        self.volatility = volatility if callable(volatility) else lambda t: float(volatility)
        self.rate_key = rate_key

    def step(self, t, x, dt, dW, context=None):
        if context is None or self.rate_key not in context:
            raise KeyError(f"Missing '{self.rate_key}' in process context")
        r = context[self.rate_key]
        sigma = self.volatility(t)
        return x * np.exp((r - 0.5*sigma**2)*dt + sigma*dW)
