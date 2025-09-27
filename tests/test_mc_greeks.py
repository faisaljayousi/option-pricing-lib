import numpy as np
import pytest

from greeks.bs_mc import (
    delta_call_bump,
    delta_call_pathwise,
    delta_put_bump,
    delta_put_pathwise,
)
from models.gbm import generate_gbm_paths, make_rng
from pricers.bs_vanilla import call_greeks, put_greeks
from pricers.mc_vanilla import price_european_vanilla_mc


def test_pathwise_delta_matches_bs_within_3sigma():
    """
    Pathwise delta estimator should match Black–Scholes delta within 3 SE.

    The pathwise estimator computes d/dS0 of the discounted payoff along 
    each simulated path. For European calls under GBM, this yields an 
    unbiased (or asymptotically unbiased) estimator of Delta, with an 
    empirically measurable standard error.

    What this test does
    -------------------
    - Simulates paths via `price_european_vanilla_mc` to reuse the same model
      and discretisation used for pricing.
    - Computes the pathwise Delta and its standard error using
      `delta_call_pathwise(paths, k, r, T)`.
    - Compares to the analytic Black–Scholes Delta from `call_greeks`.

    Pass criterion
    --------------
    |Δ_pathwise − Δ_BS| ≤ 3 × se
    """
    s0, k, r, q, sigma, T = 100, 110, 0.01, 0.005, 0.25, 1.0
    _, _, paths = price_european_vanilla_mc(
        s0, k, r, q, sigma, T, steps=252, n_paths=25000, return_paths=True, seed=777
    )
    delta_pw, se = delta_call_pathwise(paths, k, r, T)
    delta_bs = call_greeks(s0, k, r, q, sigma, T)["delta"]
    assert abs(delta_pw - delta_bs) <= 3 * se


@pytest.mark.parametrize("callput", ["call", "put"])
def test_pathwise_delta_matches_bs(callput):
    s0, k, r, q, sigma, T = 100, 100, 0.02, 0.0, 0.2, 1.0
    steps, n_paths, seed = 252, 50_000, 42

    rng = make_rng(seed)
    S_paths = generate_gbm_paths(s0, r, q, sigma, T, steps, n_paths, rng=rng)

    if callput == "call":
        delta_pw, se = delta_call_pathwise(S_paths, k, r, T)
        delta_bs = call_greeks(s0, k, r, q, sigma, T)["delta"]
    else:
        delta_pw, se = delta_put_pathwise(S_paths, k, r, T)
        delta_bs = put_greeks(s0, k, r, q, sigma, T)["delta"]

    assert abs(delta_pw - delta_bs) <= 3 * se


@pytest.mark.parametrize("callput", ["call", "put"])
def test_bump_delta_matches_bs(callput):
    s0, k, r, q, sigma, T = 100, 100, 0.02, 0.0, 0.2, 1.0
    steps, n_paths, seed = 252, 50_000, 123

    if callput == "call":
        delta_bump, se = delta_call_bump(
            s0, k, r, q, sigma, T, steps=steps, n_paths=n_paths, h=1.0, seed=seed
        )
        delta_bs = call_greeks(s0, k, r, q, sigma, T)["delta"]
    else:
        delta_bump, se = delta_put_bump(
            s0, k, r, q, sigma, T, steps=steps, n_paths=n_paths, h=1.0, seed=seed
        )
        delta_bs = put_greeks(s0, k, r, q, sigma, T)["delta"]

    assert abs(delta_bump - delta_bs) <= 3 * se

