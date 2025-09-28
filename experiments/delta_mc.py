"""
Monte Carlo convergence demo for European Greeks (Delta).

Goal
----
Estimate a European call price under GBM via Monte Carlo (MC) for increasing numbers of paths and compare against the closed-form Black–Scholes (BS) price.


What it shows
-------------
- MC estimates approach the BS price as N grows.
- The ±2·SE error band shrinks at the canonical O(1/sqrt(N)) rate.
- Reproducible results via a fixed seed.

Outputs
-------
- Interactive plot (unless --no-show) and optional PNG (--save).
- Optional CSV with N, MC estimate, SE, and absolute error (--csv).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from greeks.bs_mc import delta_call_bump, delta_call_pathwise
from pricers.bs_vanilla import call_greeks
from pricers.mc_vanilla import price_european_vanilla_mc

# ---------- Robust, OS-agnostic paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "experiments" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DELTA_TABLE_TEX = OUT_DIR / "delta_table.tex"
DELTA_CONV_PNG = OUT_DIR / "delta_convergence.png"
DELTA_COMPARISON_TEX = OUT_DIR / "delta_comparison.tex"


def run_table() -> pd.DataFrame:
    s0, r, q, sigma = 100, 0.02, 0.0, 0.2
    grid = [(80, 0.25), (100, 0.25), (120, 0.25), (100, 1.0)]
    rows = []
    for K, T in grid:
        # Analytic
        bs_delta = call_greeks(s0, K, r, q, sigma, T)["delta"]
        _, _, paths = price_european_vanilla_mc(
            s0,
            K,
            r,
            q,
            sigma,
            T,
            steps=252,
            n_paths=50_000,
            seed=123,
            return_paths=True,
        )
        delta_pw, se = delta_call_pathwise(paths, K, r, T)
        rows.append(
            dict(K=K, T=T, Delta_BS=bs_delta, Delta_MC=delta_pw, SE=se)
        )

    df = pd.DataFrame(rows)
    print(df)
    df.to_latex(DELTA_TABLE_TEX, index=False, float_format="%.4f")
    return df


def run_convergence() -> None:
    s0, k, r, q, sigma, T = 100, 100, 0.02, 0.0, 0.2, 1.0
    bs_delta = call_greeks(s0, k, r, q, sigma, T)["delta"]
    Ns = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    errors = []

    for N in Ns:
        _, _, paths = price_european_vanilla_mc(
            s0,
            k,
            r,
            q,
            sigma,
            T,
            steps=252,
            n_paths=N,
            seed=123,
            return_paths=True,
        )
        delta_pw, se = delta_call_pathwise(paths, k, r, T)
        err = abs(delta_pw - bs_delta)
        errors.append(err)
        print(
            f"N={N:6d}, Delta_MC={delta_pw:.4f}, BS={bs_delta:.4f}, |err|={err:.5f}"
        )

    plt.figure()
    plt.loglog(Ns, errors, "o-", label="MC abs error")
    plt.loglog(
        Ns,
        [errors[0] * (Ns[0] / N) ** 0.5 for N in Ns],
        "--",
        label="N^{-1/2}",
    )
    plt.xlabel("Number of paths N")
    plt.ylabel("Absolute error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DELTA_CONV_PNG, dpi=150)
    plt.close()


def run_comparison() -> pd.DataFrame:
    s0, k, r, q, sigma, T = 100, 100, 0.02, 0.0, 0.2, 1.0
    bs_delta = call_greeks(s0, k, r, q, sigma, T)["delta"]
    Ns = [2000, 5000, 10000, 20000]
    rows = []

    for N in Ns:
        _, _, paths = price_european_vanilla_mc(
            s0,
            k,
            r,
            q,
            sigma,
            T,
            steps=252,
            n_paths=N,
            seed=123,
            return_paths=True,
        )
        delta_pw, se_pw = delta_call_pathwise(paths, k, r, T)
        delta_bump, se_bump = delta_call_bump(
            s0, k, r, q, sigma, T, steps=252, n_paths=N, seed=123, h=0.01
        )
        rows.append(
            dict(
                N=N,
                Delta_BS=bs_delta,
                Delta_pw=delta_pw,
                SE_pw=se_pw,
                Delta_bump=delta_bump,
                SE_bump=se_bump,
            )
        )

    df = pd.DataFrame(rows)
    print(df)
    df.to_latex(DELTA_COMPARISON_TEX, index=False, float_format="%.4f")
    return df


def run_robustness() -> None:
    s0, k, r, q, sigma, T = 100, 100, 0.02, 0.0, 0.2, 1.0
    bs_delta = call_greeks(s0, k, r, q, sigma, T)["delta"]
    for steps in [52, 252]:
        _, _, paths = price_european_vanilla_mc(
            s0,
            k,
            r,
            q,
            sigma,
            T,
            steps=steps,
            n_paths=50_000,
            seed=42,
            return_paths=True,
        )
        delta_pw, se = delta_call_pathwise(paths, k, r, T)
        print(
            f"steps={steps}, Delta_MC={delta_pw:.4f} ± {se:.4f}, BS={bs_delta:.4f}"
        )


if __name__ == "__main__":
    run_table()
    run_convergence()
    run_comparison()
    run_robustness()
