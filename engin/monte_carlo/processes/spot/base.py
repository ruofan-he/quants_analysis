from abc import ABC, abstractmethod


class SpotProcess(ABC):
    """One-dimensional spot-price process."""

    def __init__(self, initial_value: float):
        self.initial_value = float(initial_value)

    @abstractmethod
    def step(self, t, spot, dt, dW, context=None):
        pass
