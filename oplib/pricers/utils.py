from __future__ import annotations

from typing import Tuple

import numpy as np


def _check_inputs(s: float, k: float, sigma: float, T: float) -> None:
    """
    Validate basic input domains.

    Parameters
    ----------
    s : float
        Spot price. Must be > 0.
    k : float
        Strike price. Must be > 0.
    sigma : float
        Annualised volatility. Must be >= 0.
    T : float
        Time to maturity in years. Must be >= 0.
    """
    if s <= 0.0 or k <= 0.0:
        raise ValueError("s and k must be > 0.")
    if T < 0.0:
        raise ValueError("T must be >= 0.")
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0.")


def _d1_d2(
    s: float, k: float, r: float, q: float, sigma: float, T: float
) -> Tuple[float, float]:
    """
    Compute Black–Scholes d1 and d2.

    Requires T > 0 and sigma > 0. Edge cases (T<=0 or sigma<=0) must be
    handled by the caller.
    """
    for name, v in dict(s=s, k=k, r=r, q=q, sigma=sigma, T=T).items():
        if not np.isfinite(v):
            raise ValueError(f"{name} must be finite.")
    if s <= 0.0 or k <= 0.0:
        raise ValueError("s and k must be > 0.")
    if T <= 0.0:
        raise ValueError("T must be > 0 for _d1_d2; handle T<=0 in caller.")
    if sigma <= 0.0:
        raise ValueError(
            "sigma must be > 0 for _d1_d2; handle sigma<=0 in caller."
        )

    vol_sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrtT
    d2 = d1 - vol_sqrtT
    return float(d1), float(d2)


def _disc(q: float, r: float, T: float) -> Tuple[float, float]:
    """
    Compute discount factors for dividend yield and risk-free rate.

    Parameters
    ----------
    q : float
        Continuous dividend yield (per year).
    r : float
        Continuously compounded risk-free rate (per year).
    T : float
        Time to maturity in years. Must be >= 0.

    Returns
    -------
    (dq, dr) : tuple of float
        - dq = exp(-q * T), the dividend discount factor.
        - dr = exp(-r * T), the risk-free discount factor.
    """
    return np.exp(-q * T), np.exp(-r * T)
