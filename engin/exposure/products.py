from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


class Portfolio:

    def __init__(self, products):

        if isinstance(products, dict):
            self.products = dict(products)

        else:
            self.products = {
                f"{type(p).__name__}_{i}": p
                for i, p in enumerate(products)
            }

        if not self.products:
            raise ValueError("Empty portfolio")

        for name, product in self.products.items():
            if not isinstance(product, Product):
                raise TypeError(
                    f"{name} is not a Product"
                )

    def __iter__(self):
        return iter(self.products.items())

    def __len__(self):
        return len(self.products)


@dataclass
class Cashflow:
    payment_time: float
    amount: object  # scalar or (n_paths,)


def interpolate_market(time_grid, paths, t):

    time_grid = np.asarray(time_grid)
    paths = np.asarray(paths)

    return np.array([
        np.interp(t, time_grid, path)
        for path in paths
    ])


class Product(ABC):
    def __init__(self, discount_curve="rate"):
        self.discount_curve = discount_curve

    @property
    @abstractmethod
    def market_factors(self): ...

    @abstractmethod
    def cashflows(self, market): ...


def grid_index(grid, t):
    i = int(np.argmin(np.abs(grid - t)))
    if not np.isclose(grid[i], t, atol=1e-10, rtol=0):
        raise ValueError(f"{t} is not on the simulation grid")
    return i


class Bond(Product):
    def __init__(self, notional, coupon_rate, payment_times, discount_curve="rate"):
        super().__init__(discount_curve)
        self.notional = float(notional)
        self.coupon_rate = float(coupon_rate)
        self.payment_times = tuple(map(float, payment_times))
        if not self.payment_times or any(b <= a for a,b in zip(self.payment_times, self.payment_times[1:])):
            raise ValueError("payment_times must be non-empty and strictly increasing")

    @property
    def market_factors(self):
        return ()

    def cashflows(self, market):
        out, prev = [], 0.0
        for i, t in enumerate(self.payment_times):
            amount = self.notional * self.coupon_rate * (t-prev)
            if i == len(self.payment_times)-1:
                amount += self.notional
            out.append(Cashflow(t, amount))
            prev = t
        return out


class EquityForward(Product):
    def __init__(self, quantity, delivery_price, maturity,
                 spot_factor="stock", discount_curve="rate"):
        super().__init__(discount_curve)
        self.quantity = float(quantity)
        self.delivery_price = float(delivery_price)
        self.maturity = float(maturity)
        self.spot_factor = spot_factor

    @property
    def market_factors(self):
        return (self.spot_factor,)

    def cashflows(self, market):

        S = interpolate_market(
            market["time"],
            market["paths"][self.spot_factor],
            self.maturity,
        )

        payoff = self.quantity * (
            S - self.delivery_price
        )

        return [
            Cashflow(self.maturity, payoff)
        ]


class IRS(Product):
    """OIS-like swap: floating coupon compounds simulated short rates."""
    def __init__(self, notional, fixed_rate, accrual_periods, pay_fixed=True,
                 rate_factor="rate", discount_curve="rate"):
        super().__init__(discount_curve)
        self.notional, self.fixed_rate = float(notional), float(fixed_rate)
        self.periods = tuple((float(a),float(b)) for a,b in accrual_periods)
        self.accrual_periods = self.periods
        self.pay_fixed, self.rate_factor = bool(pay_fixed), rate_factor
        if not self.periods or any(b <= a for a,b in self.periods):
            raise ValueError("Invalid accrual_periods")

    @property
    def market_factors(self):
        return (self.rate_factor,)

    def cashflows(self, market):

        grid = np.asarray(market["time"])
        rates = market["paths"][self.rate_factor]

        result = []

        for start, end in self.accrual_periods:

            # 期間内の既存グリッドを取得
            interior = grid[
                (grid > start) & (grid < end)
            ]

            # 期間境界を含む評価グリッド
            sub_grid = np.concatenate([
                [start],
                interior,
                [end],
            ])

            # 各時点の金利パスを補間
            sub_rates = np.column_stack([
                interpolate_market(
                    grid, rates, t
                )
                for t in sub_grid
            ])

            # 短期金利の積分
            integrated_rate = np.trapezoid(
                0.5 * (
                    sub_rates[:, :-1]
                    + sub_rates[:, 1:]
                ),
                x=sub_grid,
                axis=1,
            )

            floating_coupon = self.notional * (
                np.exp(integrated_rate) - 1.0
            )

            fixed_coupon = (
                self.notional
                * self.fixed_rate
                * (end - start)
            )

            if self.pay_fixed:
                net = floating_coupon - fixed_coupon
            else:
                net = fixed_coupon - floating_coupon

            result.append(
                Cashflow(end, net)
            )

        return result
