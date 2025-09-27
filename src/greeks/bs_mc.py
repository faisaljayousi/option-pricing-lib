from __future__ import annotations

import numpy as np

from pricers.mc_vanilla import price_european_vanilla_mc


def delta_call_pathwise(S_paths: np.ndarray, K: float, r: float, T: float):
    """
    Pathwise estimator of Delta (∂V/∂S0) for a European call option under GBM.

    Implements the pathwise differentiation method: for payoff 
        C = e^{-rT} max(S_T − K, 0),
    the pathwise derivative is
        Δ = e^{-rT} * 1_{S_T > K} * (S_T / S0),
    where S_T is the terminal simulated price and S0 is the initial spot.

    Parameters
    ----------
    S_paths : ndarray, shape (n_paths, n_steps + 1)
        Simulated GBM paths, including S0 in the first column and S_T in the last.
        Typically produced by a path generator such as `generate_gbm_paths`.
    K : float
        Strike price of the option.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity in years.

    Returns
    -------
    delta_est : float
        Monte Carlo estimate of Delta (sensitivity of the option price to S0).
    stderr : float
        Standard error of the Delta estimate, computed as the sample standard
        deviation of contributions divided by sqrt(n_paths).

    Notes
    -----
    * Valid for options with differentiable payoffs almost everywhere
      (vanilla calls/puts). At the kink (S_T ≈ K), the indicator introduces
      variance but does not bias the estimator.
    """
    ST = S_paths[:, -1]
    s0 = S_paths[0, 0]
    disc = np.exp(-r * T)
    contrib = disc * (ST > K).astype(float) * (ST / s0)
    return contrib.mean(), contrib.std(ddof=1) / np.sqrt(len(contrib))


def delta_put_pathwise(S_paths: np.ndarray, K: float, r: float, T: float):
    """
    Pathwise estimator of Delta (∂V/∂S0) for a European put under GBM.

    For payoff
        P = e^{-rT} max(K − S_T, 0),
    the pathwise derivative is
        Δ = - e^{-rT} · 1_{S_T < K} · (S_T / S0).

    Parameters
    ----------
    S_paths : ndarray, shape (n_paths, n_steps + 1)
        Simulated GBM paths, including S0 in the first column and S_T in the last.
    K : float
        Strike price.
    r : float
        Risk-free rate (continuous).
    T : float
        Time to maturity in years.

    Returns
    -------
    delta_est : float
        Monte Carlo estimate of Delta.
    stderr : float
        Standard error computed as `std(contrib, ddof=1) / sqrt(n_paths)`.
    """
    ST = S_paths[:, -1]
    s0 = S_paths[0, 0]
    disc = np.exp(-r * T)
    contrib = -disc * (ST < K).astype(float) * (ST / s0)
    return contrib.mean(), contrib.std(ddof=1) / np.sqrt(len(contrib))


def delta_call_bump(
    s0: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int = 252,
    n_paths: int = 100_000,
    h: float = 1e-2,
    antithetic: bool = True,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Estimate Delta of a European call using bump-and-revalue with common random numbers (CRN).

    Parameters
    ----------
    s0 : float
        Spot price.
    k : float
        Strike price.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    sigma : float
        Volatility.
    T : float
        Time to maturity (years).
    steps : int
        Number of timesteps in MC paths.
    n_paths : int
        Number of Monte Carlo paths.
    h : float
        Spot bump size.
    antithetic : bool
        Use antithetic variance reduction.
    seed : int | None
        Random seed.

    Returns
    -------
    (delta, se) : tuple[float, float]
        Estimated Delta and its standard error (from finite difference).
    """
    if seed is None:
        seed = np.random.randint(0, 2 ** 32 - 1)

    # Up bump
    mc_up, _ = price_european_vanilla_mc(
        s0 + h,
        k,
        r,
        q,
        sigma,
        T,
        steps=steps,
        n_paths=n_paths,
        call=True,
        antithetic=antithetic,
        seed=seed,
    )

    # Down bump
    mc_down, _ = price_european_vanilla_mc(
        s0 - h,
        k,
        r,
        q,
        sigma,
        T,
        steps=steps,
        n_paths=n_paths,
        call=True,
        antithetic=antithetic,
        seed=seed,
    )

    delta = (mc_up - mc_down) / (2 * h)
    se = np.sqrt(2) * (1 / (2 * h)) * (1 / np.sqrt(n_paths))
    return delta, se


def delta_put_bump(
    s0: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int = 252,
    n_paths: int = 100_000,
    h: float = 1e-2,
    antithetic: bool = True,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Delta via bump-and-revalue (central difference) for a put with CRN.

    Identical methodology to `delta_call_bump`, but values put prices on the
    up/down bumps and differences them.

    Parameters
    ----------
    s0, k, r, q, sigma, T : float
        Standard inputs.
    steps : int, default 252
        Number of time steps per simulated path.
    n_paths : int, default 100_000
        Number of Monte Carlo paths for each valuation.
    h : float, default 1e-2
        Absolute spot bump size.
    antithetic : bool, default True
        Use antithetic variates.
    seed : int | None
        Base RNG seed; reused for both bumps (CRN).

    Returns
    -------
    delta : float
        Central-difference Delta estimate for the put.
    se : float
        Heuristic standard error as in `delta_call_bump` (see its Notes for a
        more principled paired-path variance estimate).
    """
    if seed is None:
        seed = np.random.randint(0, 2 ** 32 - 1)

    mc_up, _ = price_european_vanilla_mc(
        s0 + h,
        k,
        r,
        q,
        sigma,
        T,
        steps=steps,
        n_paths=n_paths,
        call=False,
        antithetic=antithetic,
        seed=seed,
    )

    mc_down, _ = price_european_vanilla_mc(
        s0 - h,
        k,
        r,
        q,
        sigma,
        T,
        steps=steps,
        n_paths=n_paths,
        call=False,
        antithetic=antithetic,
        seed=seed,
    )

    delta = (mc_up - mc_down) / (2 * h)
    se = np.sqrt(2) * (1 / (2 * h)) * (1 / np.sqrt(n_paths))
    return delta, se
