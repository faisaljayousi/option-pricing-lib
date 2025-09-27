from __future__ import annotations

import numpy as np

from models.gbm import generate_gbm_paths, make_rng


def price_european_vanilla_mc(
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
    *,
    return_paths: bool = False,
    seed: int | None = None,
):
    """
    Monte Carlo pricer for European vanilla options under risk-neutral GBM.

    Simulates geometric Brownian motion (GBM) paths for the underlying,
    evaluates the discounted European call/put payoff at maturity, and returns
    the Monte Carlo estimator, its standard error, and (optionally) all paths.

    The risk-neutral GBM dynamics are assumed:
        dS_t = S_t[(r - q) dt + sigma dW_t],    S_0 = s0

    The discounted payoff is:
        C = e^{-rT} E[(S_T - K)^+],           for call=True
        P = e^{-rT} E[(K - S_T)^+],           for call=False

    Parameters
    ----------
    s0 : float
        Spot price at time 0 (must be > 0).
    k : float
        Strike price (must be > 0).
    r : float
        Continuously compounded risk-free rate.
    q : float
        Continuous dividend yield.
    sigma : float
        Volatility (annualised, must be >= 0).
    T : float
        Time to maturity in years (must be > 0).
    steps : int
        Number of time steps per path (>= 1). Used by `generate_gbm_paths`.
    n_paths : int
        Number of Monte Carlo paths to simulate (>= 1).
    call : bool, optional
        If True, prices a call; otherwise, a put. Default is True.
    antithetic : bool, optional
        If True, enables antithetic variates inside `generate_gbm_paths`.
        Default is True.
    return_paths : bool, optional
        If True, returns generated paths. Default is False.
    seed : int | None, optional
        Seed for the RNG created by `make_rng(seed)` to ensure reproducibility.
        Default is None.

    Returns
    -------
    price : float
        Monte Carlo estimate of the option value.
    stderr : float
        Standard error of the estimate, computed as sample std / sqrt(N),
        where N is the number of samples in `samples`. This assumes approximate
        independence of samples and large-sample normality.
    paths : (Optional) ndarray, shape (n_effective_paths, steps + 1)
        Simulated price paths including S_0 in the first column and S_T in the
        last. 
    """
    # Reproducible RNG for the whole path generation & payoff evaluation
    rng = make_rng(seed)

    # Simulate GBM price paths: shape (n_paths, steps + 1)
    S = generate_gbm_paths(
        s0, r, q, sigma, T, steps, n_paths, antithetic=antithetic, rng=rng
    )

    # Terminal prices and vanilla payoff
    ST = S[:, -1]
    payoff = (ST - k) if call else (k - ST)
    payoff = np.maximum(payoff, 0.0)

    # Risk-neutral discounting and Monte Carlo samples
    disc = np.exp(-r * T)
    samples = disc * payoff

    # MC estimator and standard error (normal approx, large N)
    price = samples.mean()
    stderr = samples.std(ddof=1) / np.sqrt(len(samples))
    return (price, stderr, S) if return_paths else (price, stderr)
