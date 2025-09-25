from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .utils import _check_inputs, _d1_d2, _disc

_EPS = 1e-12


def call_price(s, k, r, q, sigma, T):
    """
    Black–Scholes European call price.

    Parameters
    ----------
    s, k, r, q, sigma, T : float

    Returns
    -------
    float
        Call price. At expiry (T==0) returns max(s - k, 0).
    """
    _check_inputs(s, k, sigma, T)

    if T <= _EPS:
        return float(max(s - k, 0.0))

    if sigma <= _EPS:
        disc_q, disc_r = _disc(q, r, T)
        return float(max(s * disc_q - k * disc_r, 0.0))

    d1, d2 = _d1_d2(s, k, r, q, sigma, T)
    disc_q, disc_r = _disc(q, r, T)
    return float(s * disc_q * norm.cdf(d1) - k * disc_r * norm.cdf(d2))


def put_price(s, k, r, q, sigma, T):
    """
    Black–Scholes European put price (via parity).

    Uses
    ----
    P = C − S e^{-qT} + K e^{-rT}.

    Parameters
    ----------
    s, k, r, q, sigma, T : float

    Returns
    -------
    float
        Put price. At expiry (T==0) returns max(k − s, 0).
    """
    _check_inputs(s, k, sigma, T)

    if T <= _EPS:
        return float(max(k - s, 0.0))

    if sigma <= _EPS:
        disc_q, disc_r = _disc(q, r, T)
        return float(max(k * disc_r - s * disc_q, 0.0))

    c = call_price(s, k, r, q, sigma, T)
    disc_q, disc_r = _disc(q, r, T)
    return float(c - s * disc_q + k * disc_r)


def call_greeks(s, k, r, q, sigma, T):
    """
    Greeks for a European call (theta per year).

    Parameters
    ----------
    s, k, r, q, sigma, T : float

    Returns
    -------
    dict
        Keys: delta, gamma, vega, theta, rho.

    Edge cases
    ----------
    - T <= 0: returns delta in {0, 0.5, 1} depending on s vs k, others set to 0.

    Notes
    -----
    - delta = e^{-qT} N(d1)
    - gamma = e^{-qT} n(d1)/(s sigma sqrt(T))
    - vega  = s e^{-qT} n(d1) sqrt(T)
    - theta = − s e^{-qT} n(d1) sigma / (2 sqrt(T)) − r K e^{-rT} N(d2)
              + q s e^{-qT} N(d1)
    - rho   = K T e^{-rT} N(d2)
    """
    if T <= _EPS:
        delta = 1.0 if s > k else (0.5 if s == k else 0.0)
        return dict(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    if sigma <= _EPS:
        disc_q, disc_r = _disc(q, r, T)
        s_dq = s * disc_q
        if s_dq > k * disc_r:
            delta = disc_q
            theta = -q * s_dq + r * k * disc_r
            rho = k * T * disc_r
            return dict(
                delta=float(delta),
                gamma=0.0,
                vega=0.0,
                theta=float(theta),
                rho=float(rho),
            )
        elif s_dq < k * disc_r:
            delta = 0.0
        else:
            delta = 0.5 * disc_q
        return dict(delta=float(delta), gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1, d2 = _d1_d2(s, k, r, q, sigma, T)
    disc_q, disc_r = _disc(q, r, T)
    nd1 = norm.pdf(d1)

    delta = disc_q * norm.cdf(d1)
    gamma = disc_q * nd1 / (s * sigma * np.sqrt(T))
    vega = s * disc_q * nd1 * np.sqrt(T)
    theta = (
        -s * disc_q * nd1 * sigma / (2 * np.sqrt(T))
        - r * k * disc_r * norm.cdf(d2)
        + q * s * disc_q * norm.cdf(d1)
    )
    rho = k * T * disc_r * norm.cdf(d2)

    return dict(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )


def put_greeks(s, k, r, q, sigma, T):
    """
    Greeks for a European put (theta per year).

    Method
    ------
    Uses parity relations with the call Greeks and Black–Scholes formulas:

    - Δ_put = Δ_call − e^{-qT}
    - Γ_put = Γ_call
    - Vega_put = Vega_call
    - Theta_put = − s e^{-qT} φ(d1) sigma / (2 sqrt(T))
                  + q s e^{-qT} Φ(−d1) − r K e^{-rT} Φ(−d2)
    - Rho_put = − K T e^{-rT} Φ(−d2)

    Edge cases
    ----------
    - T <= 0: returns delta ∈ {−1, 0} with other Greeks 0 (ATM handled as 0 here).

    Parameters
    ----------
    s, k, r, q, sigma, T : float

    Returns
    -------
    dict
        Keys: delta, gamma, vega, theta, rho.
    """
    if T <= _EPS:
        return {
            "delta": float(-1.0 if s < k else 0.0),
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
        }

    if sigma <= _EPS:
        disc_q, disc_r = _disc(q, r, T)
        s_dq = s * disc_q
        if s_dq > k * disc_r:
            delta = 0.0
        elif s_dq < k * disc_r:
            delta = -disc_q
            theta = q * s_dq - r * k * disc_r
            rho = -k * T * disc_r
            return dict(
                delta=float(delta),
                gamma=0.0,
                vega=0.0,
                theta=float(theta),
                rho=float(rho),
            )
        else:
            delta = -0.5 * disc_q
        return dict(delta=float(delta), gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    g_c = call_greeks(s, k, r, q, sigma, T)
    d1, d2 = _d1_d2(s, k, r, q, sigma, T)
    disc_q, disc_r = _disc(q, r, T)

    delta_p = g_c["delta"] - disc_q  # Delta_put = Delta_call - e^{-qT}
    gamma_p = g_c["gamma"]  # Gamma_put = Gamma_call
    vega_p = g_c["vega"]  # Vega_put = Vega_call

    theta_p = (
        (-(s * disc_q * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)))
        + q * s * disc_q * norm.cdf(-d1)
        - r * k * disc_r * norm.cdf(-d2)
    )  # BS formula

    # Rho_put = -T * k * e^{-rT} Phi(-d2)
    rho_p = -T * k * disc_r * norm.cdf(-d2)

    return {
        "delta": float(delta_p),
        "gamma": float(gamma_p),
        "vega": float(vega_p),
        "theta": float(theta_p),
        "rho": float(rho_p),
    }
