import os
import sys

import numpy as np
from pytest import approx

from greeks.bs_mc import delta_call_pathwise
from pricers.bs_vanilla import call_greeks, call_price, put_greeks, put_price
from pricers.mc_vanilla import price_european_vanilla_mc


def test_put_call_parity():
    """
    Validate put–call parity for European options.

    For a  stock with continuous dividend yield q and risk-free rate r, 
    European call (C) and put (P) prices with the same strike K and 
    maturity T satisfy:

        C - P = S0 * exp(-q T) - K * exp(-r T)

    What this test does
    -------------------
    - Prices a call and a put via the Black–Scholes functions `call_price` and
      `put_price` with identical inputs.
    - Computes the left-hand side (LHS) and right-hand side (RHS) of the parity
      identity.
    - Asserts that the absolute difference |LHS − RHS| is at machine-precision
      scale for these inputs.

    Pass criterion
    --------------
    |(C - P) - (S0 e^{-qT} - K e^{-rT})| < 1e-10
    """
    s, k, r, q, sigma, T = 100, 100, 0.02, 0.01, 0.2, 1.0

    c = call_price(s, k, r, q, sigma, T)
    p = put_price(s, k, r, q, sigma, T)

    lhs = c - p
    rhs = s * np.exp(-q * T) - k * np.exp(-r * T)

    assert lhs == approx(rhs, rel=1e-12, abs=1e-12), f"Parity diff={abs(lhs-rhs):.3e}"


def test_mc_price_matches_bs_within_3sigma():
    """
    Monte Carlo price should agree with Black–Scholes within 3 standard errors.

    Under Black–Scholes dynamics, a risk-neutral Monte Carlo (MC) estimator of
    a European call price is unbiased and (by the CLT) approximately normal
    around the true price with standard error `se`.

    What this test does
    -------------------
    - Computes the analytic Black–Scholes call price (`bs`).
    - Estimates the price via MC using `price_european_vanilla_mc`, which
      returns (estimate, standard_error, simulated_paths).
    - Checks that the absolute deviation |MC − BS| is ≤ 3 * se (≈ 99.7% CL
      under normality).

    Pass criterion
    --------------
    |MC − BS| ≤ 3 × se
    """
    s0, k, r, q, sigma, T = 100, 100, 0.02, 0.01, 0.2, 1.0
    bs = call_price(s0, k, r, q, sigma, T)
    mc, se = price_european_vanilla_mc(
        s0, k, r, q, sigma, T, steps=252, n_paths=20000, seed=123
    )
    assert abs(mc - bs) <= 3 * se
