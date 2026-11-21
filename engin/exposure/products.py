from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class Cashflow:
    payment_time: float
    amount: object  # scalar or (n_paths,)

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
        i = grid_index(market["time"], self.maturity)
        s = market["paths"][self.spot_factor][:, i]
        return [Cashflow(self.maturity, self.quantity*(s-self.delivery_price))]


class IRS(Product):
    """OIS-like swap: floating coupon compounds simulated short rates."""
    def __init__(self, notional, fixed_rate, accrual_periods, pay_fixed=True,
                 rate_factor="rate", discount_curve="rate"):
        super().__init__(discount_curve)
        self.notional, self.fixed_rate = float(notional), float(fixed_rate)
        self.periods = tuple((float(a),float(b)) for a,b in accrual_periods)
        self.pay_fixed, self.rate_factor = bool(pay_fixed), rate_factor
        if not self.periods or any(b <= a for a,b in self.periods):
            raise ValueError("Invalid accrual_periods")

    @property
    def market_factors(self):
        return (self.rate_factor,)

    def cashflows(self, market):
        grid, r = market["time"], market["paths"][self.rate_factor]
        out = []
        for start, end in self.periods:
            i0, i1 = grid_index(grid,start), grid_index(grid,end)
            dts = np.diff(grid[i0:i1+1])
            growth = np.prod(1.0 + r[:,i0:i1]*dts[None,:], axis=1)
            floating = self.notional*(growth-1.0)
            fixed = self.notional*self.fixed_rate*(end-start)
            net = floating-fixed if self.pay_fixed else fixed-floating
            out.append(Cashflow(end, net))
        return out
