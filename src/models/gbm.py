from __future__ import annotations

from typing import Union

import numpy as np
from numpy.random import PCG64, Generator


def make_rng(seed: int | None = None) -> Generator:
    """
    Create a NumPy Generator with PCG64 bit generator.

    Parameters
    ----------
    seed
        Optional seed for reproducibility.

    Returns
    -------
    numpy.random.Generator
    """
    return Generator(PCG64(seed))


def generate_gbm_paths(
    s0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int,
    n_paths: int,
    *,
    antithetic: bool = True,
    rng: Generator | None = None,
    return_timegrid: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Generate exact GBM sample paths under Q using log-space accumulation.

    Model:
        log S_{t+Δ} - log S_t = (r - q - 0.5 σ^2) Δ + σ sqrt(Δ) Z,   Z ~ N(0, 1)

    Parameters
    ----------
    s0 : float
        Initial spot (> 0).
    r, q : float
        Risk-free rate and dividend yield.
    sigma : float
        Volatility (>= 0).
    T : float
        Maturity in years (> 0).
    steps : int
        Number of time steps (>= 1).
    n_paths : int
        Number of paths (>= 1).
    antithetic : bool, default True
        Use antithetic variates (last path unpaired if n_paths is odd).
    rng : numpy.random.Generator, optional
        RNG to use. If None, a new PCG64-based generator is created.

    Returns
    -------
    S : (n_paths, steps+1) array
        Simulated paths with S[:, 0] = s0.
    t : (steps+1,) array
        Returned only if `return_timegrid=True`.
    """
    # Basic validation (lightweight, reuse your GBMParams.validate if desired)
    if not np.isfinite(s0) or s0 <= 0:
        raise ValueError("s0 must be a finite positive float.")
    for name, val in (("r", r), ("q", q), ("sigma", sigma), ("T", T)):
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite.")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be an integer >= 1.")
    if not isinstance(n_paths, int) or n_paths < 1:
        raise ValueError("n_paths must be an integer >= 1.")
    if T <= 0:
        raise ValueError("T must be > 0.")

    if rng is None:
        rng = Generator(PCG64())

    dt = T / steps
    mu_dt = (r - q - 0.5 * sigma * sigma) * dt
    vol_dt = sigma * np.sqrt(dt)

    # Deterministic case: no randomness, S_t = s0 * exp((r - q) t)
    if sigma == 0.0:
        t = np.arange(steps + 1, dtype=float) * dt
        path = s0 * np.exp((r - q) * t)
        S = np.repeat(path[None, :], n_paths, axis=0)
        return (S, t) if return_timegrid else S

    # Random case
    if antithetic:
        half = (n_paths + 1) // 2
        Z = rng.standard_normal((half, steps))
        Z = np.vstack([Z, -Z])[:n_paths]  # keep pairs; odd path remains unpaired
    else:
        Z = rng.standard_normal((n_paths, steps))

    # Log-space accumulation
    logS = np.empty((n_paths, steps + 1), dtype=float)
    logS[:, 0] = np.log(s0)

    # cumulative sum of log-increments, then shift by log(s0)
    np.cumsum(mu_dt + vol_dt * Z, axis=1, out=logS[:, 1:])
    logS[:, 1:] += logS[:, [0]]

    if return_timegrid:
        t = np.arange(steps + 1) * dt
        return np.exp(logS), t

    return np.exp(logS)
