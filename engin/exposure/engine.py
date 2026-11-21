import numpy as np


class ExposureEngine:
    """
    Portfolio-level pathwise exposure evaluator.

    Output:
        {
            "notice_grid": (n_notice,),
            "portfolio": {...},
            "products": {product_name: {...}}
        }
    """

    def __init__(
        self,
        market_result,
        initial_discount_factors=None,
    ):
        self.market = market_result
        self.time = np.asarray(market_result["time"], dtype=float)
        self.paths = market_result["paths"]

        self.default_df = np.asarray(
            market_result["discount_factors"], dtype=float
        )

        # Optional initial curves:
        # {"rate": (curve_times, curve_dfs), ...}
        self.initial_discount_factors = (
            initial_discount_factors or {}
        )

        if self.time.ndim != 1 or np.any(np.diff(self.time) <= 0):
            raise ValueError("Invalid simulation time grid")

        expected_shape = (
            self.default_df.shape[0],
            len(self.time),
        )

        if self.default_df.shape != expected_shape:
            raise ValueError("Invalid discount factor shape")

        if np.any(self.default_df <= 0):
            raise ValueError("Discount factors must be positive")

        self.n_paths = self.default_df.shape[0]

        for name, path in self.paths.items():
            if np.asarray(path).shape != expected_shape:
                raise ValueError(
                    f"Invalid path shape: {name}"
                )

    # --------------------------------------------------
    # Interpolation
    # --------------------------------------------------

    def interpolate(self, values, t, log=False):
        """
        Vectorized linear interpolation.

        values: (n_paths, n_grid)
        returns: (n_paths,)
        """
        values = np.asarray(values, dtype=float)

        if values.shape != (self.n_paths, len(self.time)):
            raise ValueError("Invalid interpolation input shape")

        if t < self.time[0] - 1e-12 or t > self.time[-1] + 1e-12:
            raise ValueError(f"Time {t} outside simulation range")

        # Find surrounding grid points
        idx = np.searchsorted(self.time, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.time) - 2)

        t0 = self.time[idx]
        t1 = self.time[idx + 1]

        weight = (t - t0) / (t1 - t0)

        v0 = values[:, idx]
        v1 = values[:, idx + 1]

        if log:
            if np.any(values <= 0):
                raise ValueError("Log interpolation requires positive values")

            result = np.exp(
                np.log(v0) * (1.0 - weight)
                + np.log(v1) * weight
            )
        else:
            result = v0 * (1.0 - weight) + v1 * weight

        return result

    # --------------------------------------------------
    # Discount factor
    # --------------------------------------------------

    def get_df_paths(self, curve_name):
        by_curve = self.market.get(
            "discount_factors_by_curve", {}
        )

        return np.asarray(
            by_curve.get(curve_name, self.default_df),
            dtype=float,
        )

    def get_initial_df(self, curve_name, t, df_paths):
        """
        Deterministic initial DF.

        Priority:
        1. Explicit initial market curve
        2. Cross-path average of simulated DF
        """

        curve = self.initial_discount_factors.get(curve_name)

        if curve is not None:
            curve_times, curve_dfs = curve

            curve_times = np.asarray(curve_times, dtype=float)
            curve_dfs = np.asarray(curve_dfs, dtype=float)

            if np.any(curve_dfs <= 0):
                raise ValueError("Initial DFs must be positive")

            if t < curve_times[0] or t > curve_times[-1]:
                raise ValueError(
                    f"Initial curve does not cover {t}"
                )

            return float(
                np.exp(
                    np.interp(
                        t,
                        curve_times,
                        np.log(curve_dfs),
                    )
                )
            )

        # MC estimate of initial DF.
        return float(
            np.mean(
                self.interpolate(df_paths, t, log=True)
            )
        )

    # --------------------------------------------------
    # Product valuation
    # --------------------------------------------------

    def evaluate_product(self, product, notice_grid):

        for factor in product.market_factors:
            if factor not in self.paths:
                raise KeyError(
                    f"Missing market factor: {factor}"
                )

        df = self.get_df_paths(product.discount_curve)

        if df.shape != self.default_df.shape:
            raise ValueError("Invalid discount factor path shape")

        if np.any(df <= 0):
            raise ValueError("Discount factors must be positive")

        cashflows = list(product.cashflows(self.market))

        values = np.zeros(
            (self.n_paths, len(notice_grid))
        )

        for j, t in enumerate(notice_grid):

            # ------------------------------------------
            # Initial valuation
            # ------------------------------------------

            if np.isclose(t, 0.0, atol=1e-12):

                pv = 0.0

                for cf in cashflows:

                    if cf.payment_time <= 0:
                        continue

                    amount = np.asarray(
                        cf.amount, dtype=float
                    )

                    if amount.ndim == 0:
                        expected_amount = float(amount)

                    elif amount.shape == (self.n_paths,):
                        expected_amount = float(
                            np.mean(amount)
                        )

                    else:
                        raise ValueError(
                            "Invalid cashflow shape"
                        )

                    df0t = self.get_initial_df(
                        product.discount_curve,
                        cf.payment_time,
                        df,
                    )

                    pv += expected_amount * df0t

                # Same initial value across every path.
                values[:, j] = pv
                continue

            # ------------------------------------------
            # Future valuation
            # ------------------------------------------

            df_t = self.interpolate(
                df, t, log=True
            )

            for cf in cashflows:

                if cf.payment_time <= t + 1e-12:
                    continue

                df_payment = self.interpolate(
                    df,
                    cf.payment_time,
                    log=True,
                )

                amount = np.asarray(
                    cf.amount, dtype=float
                )

                if amount.ndim == 0:
                    amount = np.full(
                        self.n_paths, float(amount)
                    )

                if amount.shape != (self.n_paths,):
                    raise ValueError(
                        "Cashflow must be scalar or (n_paths,)"
                    )

                values[:, j] += (
                    amount * df_payment / df_t
                )

        return values

    # --------------------------------------------------
    # Portfolio valuation
    # --------------------------------------------------

    def evaluate(self, portfolio, notice_grid):

        from .products import Portfolio

        if not isinstance(portfolio, Portfolio):
            portfolio = Portfolio(portfolio)

        notice_grid = np.asarray(
            notice_grid, dtype=float
        )

        if notice_grid.ndim != 1:
            raise ValueError(
                "notice_grid must be one-dimensional"
            )

        if (
            np.any(notice_grid < self.time[0])
            or np.any(notice_grid > self.time[-1])
        ):
            raise ValueError(
                "Notice grid outside simulation range"
            )

        product_values = {}

        for name, product in portfolio:

            product_values[name] = self.evaluate_product(
                product,
                notice_grid,
            )

        # Netting at portfolio level.
        portfolio_value = np.sum(
            list(product_values.values()),
            axis=0,
        )

        return {
            "notice_grid": notice_grid,

            "portfolio": self.exposure_statistics(
                portfolio_value
            ),

            "products": {
                name: self.exposure_statistics(value)
                for name, value in product_values.items()
            },
        }

    @staticmethod
    def exposure_statistics(value):

        positive = np.maximum(value, 0.0)
        negative = np.maximum(-value, 0.0)

        return {
            "value": value,
            "positive_exposure": positive,
            "negative_exposure": negative,
            "EE": np.mean(positive, axis=0),
            "ENE": np.mean(negative, axis=0),
            "mean_value": np.mean(value, axis=0),
        }