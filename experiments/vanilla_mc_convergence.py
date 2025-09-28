"""
Convergence experiment with common random numbers.

Plots |MC - BS| vs N on log-log axes, alongside the MC standard error curve.

Usage
-----
PYTHONPATH=src python experiments/vanilla_mc_convergence.py \
    --outdir experiments/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pricers.bs_vanilla import call_price


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MC convergence with nested sampling (CRN)."
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for artifacts (default: <project_root>/experiments/results)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    SCRIPT_DIR = Path(__file__).resolve().parent

    # Create a "results" folder inside the same folder as this script
    outdir = SCRIPT_DIR / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "convergence_vanilla_mc.png"

    # --- Params ---
    S0, K = 100.0, 100.0
    r, q, sigma, T = 0.02, 0.0, 0.2, 1.0
    max_N = 200_000
    Ns = np.array(
        [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]
    )

    # --- Benchmark (analytic BS) ---
    bs_val = call_price(S0, K, r, q, sigma, T)

    # --- One RNG stream; nested samples (CRN) ---
    rng = np.random.default_rng(20250925)
    Z = rng.standard_normal(max_N)

    drift = (r - q - 0.5 * sigma**2) * T
    volT = sigma * np.sqrt(T)
    ST = S0 * np.exp(drift + volT * Z)
    X = np.exp(-r * T) * np.maximum(ST - K, 0.0)  # discounted payoffs

    means = np.array([X[:n].mean() for n in Ns])
    ses = np.array([X[:n].std(ddof=1) / np.sqrt(n) for n in Ns])
    errs = np.abs(means - bs_val)

    # --- Print a small table ---
    for n, m, se, e in zip(Ns, means, ses, errs):
        print(
            f"N={n:7d} | MC={m:10.6f} | SE={se:9.6f} | BS={bs_val:10.6f} | abs err={e:9.6f}"
        )

    # --- Plot ---
    plt.figure(figsize=(6, 4))
    plt.loglog(Ns, errs, "o-", label="|MC - BS| (nested)")
    plt.loglog(Ns, ses, "s--", label="MC standard error")
    # reference slope: anchor at first point of SE
    ref = ses[0] * (Ns[0] / Ns) ** 0.5
    plt.loglog(Ns, ref, "k:", label=r"$\propto N^{-1/2}$")

    plt.xlabel("Number of paths (N)")
    plt.ylabel("Error")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved plot to {outfile}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
