import numpy as np
from exposure import Bond, EquityForward, IRS, ExposureEngine, Portfolio

# Normally returned by MonteCarloEngine.simulate()
n, grid = 5000, np.linspace(0, 5, 501)
dt = grid[1]-grid[0]
rng = np.random.default_rng(7)
r = np.full((n,len(grid)), 0.02)
s = 100*np.exp(np.cumsum(-0.5*0.2**2*dt + 0.2*np.sqrt(dt)*
                         rng.standard_normal((n,len(grid))), axis=1))
df = np.exp(-np.cumsum(r*dt, axis=1))
df[:,0] = 1.0
market = {"time":grid, "paths":{"rate":r,"stock":s}, "discount_factors":df}

products = [
    Bond(1_000_000, 0.03, [1,2,3,4,5]),
    EquityForward(100, 100, 5, spot_factor="stock"),
    IRS(1_000_000, 0.025, [(0,1),(1,2),(2,3),(3,4),(4,5)]),
]

portfolio = Portfolio({
    "bond_1": Bond(
        notional=1_000_000,
        coupon_rate=0.03,
        payment_times=[0.75, 1.5, 2.25, 3.0, 5.0],
    ),

    "equity_forward_1": EquityForward(
        quantity=100,
        delivery_price=100,
        maturity=4.73,
        spot_factor="stock",
    ),

    # "irs_1": IRS(
    #     notional=1_000_000,
    #     fixed_rate=0.025,
    #     accrual_periods=[
    #         (0, 1), (1, 2), (2, 3),
    #         (3, 4), (4, 5),
    #     ],
    #     rate_factor="rate",
    # ),
})


notice = [0,1,2,3,4]
evaluator = ExposureEngine(market)
result = evaluator.evaluate(portfolio, notice)
