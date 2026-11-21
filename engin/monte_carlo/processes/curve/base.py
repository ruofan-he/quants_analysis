from abc import ABC, abstractmethod
import numpy as np


class CurveProcess(ABC):
    """One-dimensional curve factor process (e.g. short rate or hazard rate)."""

    def __init__(self, initial_value: float):
        self.initial_value = float(initial_value)

    @abstractmethod
    def step(self, t, x, dt, dW, context=None):
        """Advance factor state by one time step."""

    def discount_rate(self, x):
        """Rate used to accumulate a discount/survival factor."""
        return x
