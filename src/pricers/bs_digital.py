from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .utils import _check_inputs, _d1_d2


def digital_call_price(s, k, r, q, sigma, T):
    """
    Cash-or-nothing binary (digital) call price.

    Formula
    -------
    e^{-rT} N(d2)  (when T>0 and sigma>0)

    Parameters
    ----------
    s, k, r, q, sigma, T : float

    Returns
    -------
    float
        Binary call price.
    """
    _check_inputs(s, k, sigma, T)

    if T == 0:
        return float(np.heaviside(s - k, 0.5))
    if sigma == 0.0:
        ST = s * np.exp((r - q) * T)
        payoff_prob = 1.0 if ST > k else (0.5 if ST == k else 0.0)
        return float(np.exp(-r * T) * payoff_prob)

    _, d2 = _d1_d2(s, k, r, q, sigma, T)
    return np.exp(-r * T) * norm.cdf(d2)


def digital_put_price(s, k, r, q, sigma, T):
    """
    Cash-or-nothing binary (digital) put price.

    Formula
    -------
    e^{-rT} N(−d2)  (when T>0 and sigma>0)

    Parameters
    ----------
    s, k, r, q, sigma, T : float

    Returns
    -------
    float
        Digital put price.

    Warning
    -------
    The tie-break convention at expiry differs from `binary_call_price`:
    here T<=0 returns {0 or 1} without a 0.5 tie. Align if you require symmetry.
    """
    if T <= 0:
        return float(1.0 if s < k else 0.0)

    _, d2 = _d1_d2(s, k, r, q, sigma, T)
    return np.exp(-r * T) * norm.cdf(-d2)
