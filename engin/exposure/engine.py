import numpy as np

class ExposureEngine:
    """
    Pathwise discounted-realized-cashflow evaluator.
    Result value shape = (n_paths, n_notice_dates).
    """
    def __init__(self, market_result):
        self.market = market_result
        self.time = np.asarray(market_result["time"], dtype=float)
        self.paths = market_result["paths"]
        self.default_df = np.asarray(market_result["discount_factors"], dtype=float)

    def _idx(self, t):
        i = int(np.argmin(np.abs(self.time-t)))
        if not np.isclose(self.time[i], t, atol=1e-10, rtol=0):
            raise ValueError(f"{t} is not on the simulation grid")
        return i

    def evaluate(self, product, notice_grid):
        notice_grid = np.asarray(notice_grid, dtype=float)
        if notice_grid.ndim != 1:
            raise ValueError("notice_grid must be one-dimensional")
        for factor in product.market_factors:
            if factor not in self.paths:
                raise KeyError(f"Missing required factor: {factor}")

        # Optional extension: separate simulated DF paths by curve name.
        df = self.market.get("discount_factors_by_curve", {}).get(
            product.discount_curve, self.default_df
        )
        n_paths = self.default_df.shape[0]
        cfs = list(product.cashflows(self.market))
        values = np.zeros((n_paths, len(notice_grid)))

        for j, t in enumerate(notice_grid):
            it = self._idx(t)
            df_t = df[:,it]
            if np.any(df_t <= 0):
                raise ValueError("Discount factors must be positive")
            for cf in cfs:
                if cf.payment_time <= t + 1e-12:
                    continue
                ip = self._idx(cf.payment_time)
                amount = np.asarray(cf.amount, dtype=float)
                if amount.ndim == 0:
                    amount = np.full(n_paths, float(amount))
                if amount.shape != (n_paths,):
                    raise ValueError("Cashflow amount must be scalar or (n_paths,)")
                values[:,j] += amount * df[:,ip] / df_t

        return {
            "notice_grid": notice_grid,
            "value": values,
            "positive_exposure": np.maximum(values, 0.0),
            "negative_exposure": np.maximum(-values, 0.0),
        }
