import numpy as np

from monte_carlo import (
    MonteCarloEngine,
    HullWhiteProcess,
    HullWhiteTheta,
    BlackScholesProcess,
)

# Market data / model parameters are assembled outside the engine.
times = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
dfs = [1.0, 0.99, 0.975, 0.94, 0.85, 0.70]

alpha_r = 0.1
sigma_r = 0.01
theta_model = HullWhiteTheta(times, dfs, alpha_r, sigma_r)

rate = HullWhiteProcess(
    initial_value=0.01,
    alpha=alpha_r,
    sigma=sigma_r,
    theta=theta_model.theta,
)
stock = BlackScholesProcess(
    initial_value=100.0,
    volatility=0.20,
    rate_key="rate",
)
hazard = HullWhiteProcess(
    initial_value=0.02,
    alpha=0.3,
    sigma=0.005,
    theta=lambda t: 0.3 * 0.02,
)

# Factor order is the insertion order of the processes dictionary:
# stock, rate, hazard.
rho_sr, rho_sl, rho_rl = 0.2, 0.1, 0.35
corr = np.array([
    [1.0,    rho_sr, rho_sl],
    [rho_sr, 1.0,    rho_rl],
    [rho_sl, rho_rl, 1.0],
])

engine = MonteCarloEngine(
    processes={"stock": stock, "rate": rate, "hazard": hazard},
    correlation=corr,
    maturity=5.0,
    n_steps=250,
    n_paths=20_000,
    seed=42,
    discount_curve_key="rate",
)

result = engine.simulate()
S = result["paths"]["stock"]
r = result["paths"]["rate"]
hazard_rate = result["paths"]["hazard"]
df = result["discount_factors"]

call_price = np.mean(df[:, -1] * np.maximum(S[:, -1] - 100.0, 0.0))
print("Call price:", call_price)

# Hull-White zero-coupon bond values at each simulation time.
t = result["time"]
zcb = np.column_stack([
    rate.zcb(float(ti), r[:, i], maturity=5.0)
    for i, ti in enumerate(t)
])
print("Mean ZCB at t=0:", zcb[:, 0].mean())
