import numpy as np
from ..random.correlated import CorrelatedBrownian
from scipy.integrate import trapezoid


class MonteCarloEngine:
    """
    Generic simulation orchestrator.

    Processes own their dynamics. The engine owns the time grid, random shocks,
    state-path allocation, process-to-factor mapping and discount accumulation.
    """

    def __init__(
        self,
        processes,
        correlation,
        maturity,
        n_steps,
        n_paths,
        seed=None,
        discount_curve_key=None,
    ):
        self.processes = dict(processes)
        if not self.processes:
            raise ValueError("At least one process is required")

        self.maturity = float(maturity)
        self.n_steps = int(n_steps)
        self.n_paths = int(n_paths)
        if self.n_steps <= 0 or self.n_paths <= 0 or self.maturity <= 0:
            raise ValueError("maturity, n_steps and n_paths must be positive")

        self.dt = self.maturity / self.n_steps
        self.time_grid = np.linspace(0.0, self.maturity, self.n_steps + 1)
        self.random = CorrelatedBrownian(list(self.processes), correlation, seed)
        self.discount_curve_key = discount_curve_key

        if discount_curve_key is not None and discount_curve_key not in self.processes:
            raise KeyError(f"Unknown discount curve key: {discount_curve_key}")

    def simulate(self):
        paths = {
            name: np.empty((self.n_paths, self.n_steps + 1), dtype=float)
            for name in self.processes
        }
        for name, process in self.processes.items():
            paths[name][:, 0] = process.initial_value

        discount_factors = None
        if self.discount_curve_key is not None:
            discount_factors = np.ones((self.n_paths, self.n_steps + 1))

        for i, t in enumerate(self.time_grid[:-1]):
            dW = self.random.generate(self.n_paths, self.dt)

            # Context is the collection of current factor values.
            context = {name: paths[name][:, i] for name in self.processes}

            for j, (name, process) in enumerate(self.processes.items()):
                paths[name][:, i + 1] = process.step(
                    t=t,
                    x=paths[name][:, i],
                    dt=self.dt,
                    dW=dW[:, j],
                    context=context,
                )

            if discount_factors is not None:
                rate1 = paths[self.discount_curve_key][:, i]
                t1 = self.time_grid[i]
                t2 = self.time_grid[i + 1]
                rate = self.processes[self.discount_curve_key].discount_rate(t1, t2, rate1)
                discount_factors[:, i + 1] = (
                    discount_factors[:, i] * np.exp(-rate*self.dt)
                )


        result = {"time": self.time_grid, "paths": paths}
        if discount_factors is not None:
            result["discount_factors"] = discount_factors
        return result
