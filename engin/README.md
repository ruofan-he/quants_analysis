# Monte Carlo process refactor

## Install
```bash
pip install numpy scipy
```

Run the example from the directory containing `example.py`:
```bash
python example.py
```

## Design
- `processes/curve`: short-rate / hazard-rate factors and curve-model utilities.
- `processes/spot`: spot-price dynamics.
- `random`: correlated Brownian increments.
- `engine`: time stepping, path allocation and discount-factor accumulation.

The engine does not contain Hull-White or Black-Scholes equations. It receives
process instances and supplies each process with its Brownian increment and the
current state context.

The example is a structural starting point. In production, add validation for
the market-data domain, time-dependent correlation, non-uniform time grids,
variance reduction, and model-specific exact discretizations as needed.
