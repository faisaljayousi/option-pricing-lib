import os
import sys

import numpy as np

from pricers.bs_digital import digital_call_price, digital_put_price
from pricers.bs_vanilla import call_price, put_price
from pricers.mc_digital import price_digital_mc
from pricers.mc_vanilla import price_european_vanilla_mc


def test_digital_put_vs_call_relation():
    s, k, r, q, sigma, T = 100, 95, 0.02, 0.0, 0.25, 0.5
    c = digital_call_price(s, k, r, q, sigma, T)
    p = digital_put_price(s, k, r, q, sigma, T)

    # Phi(d2) + Phi(-d2) = 1 --> c + p = e^{-rT}
    assert abs((c + p) - np.exp(-r * T)) < 1e-10


def test_digital_equals_minus_dC_dK():
    s0, r, q, sigma, T = 100, 0.015, 0.0, 0.2, 1.0
    k, eps = 100.0, 1e-3
    # finite-diff in strike
    dC_dK = (
        call_price(s0, k + eps, r, q, sigma, T)
        - call_price(s0, k - eps, r, q, sigma, T)
    ) / (2 * eps)
    digi = digital_call_price(s0, k, r, q, sigma, T)
    assert abs(digi + dC_dK) < 5e-4


def test_digital_call_mc_matches_bs_within_3sigma():
    s0, k, r, q, sigma, T = 100, 100, 0.02, 0.0, 0.2, 1.0
    # analytic
    bs_val = digital_call_price(s0, k, r, q, sigma, T)
    # MC
    mc_val, se, _ = price_digital_mc(
        s0, k, r, q, sigma, T, steps=252, n_paths=50_000, call=True, seed=123
    )
    assert abs(mc_val - bs_val) <= 3 * se
