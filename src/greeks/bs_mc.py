from __future__ import annotations

import numpy as np


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

