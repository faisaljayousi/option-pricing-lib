from __future__ import annotations
import numpy as np
from scipy.stats import norm


def geometric_asian_call_bs(
    s0: float, k: float, r: float, q: float, sigma: float, T: float, m: int
) -> float:
    """
    Black–Scholes closed-form price for a geometric Asian call.

    Model
    -----
    Under GBM: S_t = S_0 * exp((r - q - 0.5*sigma^2)t + sigma*W_t).
    For equally spaced fixings 0 < t_1 < ... < t_m = T, the geometric average
        G = (Π_{j=1}^m S_{t_j})^{1/m}
    is lognormal: log G ~ N(mu_G, v_G).

    Parameters
    ----------
    s0 : float
        Initial spot S_0 (> 0).
    k : float
        Strike K (> 0).
    r : float
        Risk-free rate (continuous compounding).
    q : float
        Continuous dividend yield.
    sigma : float
        Volatility (annualised, >= 0).
    T : float
        Time to maturity in years (>= 0).
    m : int
        Number of fixings (positive integer).

    Returns
    -------
    float
        Present value of the geometric Asian call.

    Edge cases
    ----------
    - If T <= 0, returns max(s0 - k, 0) (payoff at “maturity”).
    - Raises ValueError if m <= 0.
    """
    if T <= 0:
        return max(s0 - k, 0.0)

    m = int(m)
    if m <= 0:
        raise ValueError("m (number of fixings) must be a positive integer.")

    mu_G = np.log(s0) + (r - q - 0.5 * sigma ** 2) * T * (m + 1) / (2.0 * m)
    v_G = (sigma ** 2) * T * ((m + 1) * (2.0 * m + 1.0)) / (6.0 * m ** 2)

    sqrt_v = np.sqrt(v_G)
    lnK = np.log(k)

    d2 = (mu_G - lnK) / sqrt_v
    d1 = d2 + sqrt_v

    EX = np.exp(mu_G + 0.5 * v_G)  # E[G]
    price = np.exp(-r * T) * (EX * norm.cdf(d1) - k * norm.cdf(d2))
    return float(price)


def geometric_asian_put_bs(
    s0: float, k: float, r: float, q: float, sigma: float, T: float, m: int
) -> float:
    """
    Black–Scholes closed-form price for a geometric Asian put.

    The distribution of the geometric average G is the same as in the call case:
    log G ~ N(mu_G, v_G). For a put payoff (K - G)^+ one gets the standard
    lognormal put formula with mean parameter E[G] = exp(mu_G + 0.5 v_G).

    Parameters
    ----------
    s0 : float
        Initial spot S_0 (> 0).
    k : float
        Strike K (> 0).
    r : float
        Risk-free rate (continuous compounding).
    q : float
        Continuous dividend yield.
    sigma : float
        Volatility (annualised, >= 0).
    T : float
        Time to maturity in years (>= 0).
    m : int
        Number of fixings (positive integer).

    Returns
    -------
    float
        Present value of the geometric Asian put.

    Edge cases
    ----------
    - If T <= 0, returns max(k - s0, 0) (payoff at “maturity”).
    - Raises ValueError if m <= 0.
    """
    if T <= 0:
        return max(k - s0, 0.0)

    m = int(m)
    if m <= 0:
        raise ValueError("m (number of fixings) must be a positive integer.")

    mu_G = np.log(s0) + (r - q - 0.5 * sigma ** 2) * T * (m + 1) / (2.0 * m)
    v_G = (sigma ** 2) * T * ((m + 1) * (2.0 * m + 1.0)) / (6.0 * m ** 2)

    sqrt_v = np.sqrt(v_G)
    lnK = np.log(k)

    d2 = (mu_G - lnK) / sqrt_v
    d1 = d2 + sqrt_v

    EX = np.exp(mu_G + 0.5 * v_G)  # E[G]
    price = np.exp(-r * T) * (k * norm.cdf(-d2) - EX * norm.cdf(-d1))
    return float(price)
