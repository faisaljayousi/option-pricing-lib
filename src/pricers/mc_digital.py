from __future__ import annotations

import numpy as np

from models.gbm import generate_gbm_paths, make_rng


def price_digital_mc(
    s0: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int,
    n_paths: int,
    call: bool = True,
    antithetic: bool = True,
    seed: int | None = None,
):
    rng = make_rng(seed)
    S = generate_gbm_paths(
        s0, r, q, sigma, T, steps, n_paths, antithetic=antithetic, rng=rng
    )
    ST = S[:, -1]
    indicator = (ST > k) if call else (k > ST)
    payoff = indicator.astype(float)
    disc = np.exp(-r * T)
    samples = disc * payoff
    return samples.mean(), samples.std(ddof=1) / np.sqrt(len(samples)), S
