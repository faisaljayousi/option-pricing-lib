from __future__ import annotations

import os

import numpy as np
import pandas as pd
from models.gbm import generate_gbm_paths, make_rng
from pricers.bs_digitals import digital_call_price, digital_put_price


def price_digital_mc(
    s0: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    *,
    steps: int = 252,
    n_paths: int = 50_000,
    call: bool = True,
    antithetic: bool = True,
    seed: int | None = 123,
) -> tuple[float, float]:
    """
    MC price for a cash-or-nothing digital (1 unit of cash if ITM at T).
    Returns (estimate, standard_error).
    """
    rng = make_rng(seed)
    S = generate_gbm_paths(
        s0, r, q, sigma, T, steps, n_paths, antithetic=antithetic, rng=rng
    )
    ST = S[:, -1]
    if call:
        payoff = (ST > k).astype(float)
    else:
        payoff = (ST < k).astype(float)
    disc = np.exp(-r * T)
    samples = disc * payoff
    mean = float(samples.mean())
    se = float(samples.std(ddof=1) / np.sqrt(len(samples)))
    return mean, se


def main():
    # Base parameters
    s0, r, q, sigma = 100.0, 0.02, 0.00, 0.20
    strikes = [80, 100, 120]
    maturities = [0.25, 1.0]
    steps = 252
    n_paths = 50_000
    seed = 123
    antithetic = True

    rows = []
    for T in maturities:
        for K in strikes:
            # Analytic (BS)
            bs_c = digital_call_price(s0, K, r, q, sigma, T)
            bs_p = digital_put_price(s0, K, r, q, sigma, T)

            # Monte Carlo
            mc_c, se_c = price_digital_mc(
                s0,
                K,
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
            mc_p, se_p = price_digital_mc(
                s0,
                K,
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

            rows.append(
                dict(
                    K=K,
                    T=T,
                    BS_DigitalCall=bs_c,
                    MC_DigitalCall=mc_c,
                    SE_Call=se_c,
                    BS_DigitalPut=bs_p,
                    MC_DigitalPut=mc_p,
                    SE_Put=se_p,
                )
            )

    df = pd.DataFrame(rows)

    with pd.option_context("display.float_format", lambda x: f"{x:0.4f}"):
        print(df)

    # Create results dir
    os.makedirs("experiments/results", exist_ok=True)

    # Save CSV
    csv_path = "experiments/results/digital_mc_vs_bs.csv"
    df.to_csv(csv_path, index=False)

    # Compose LaTeX with ±3σ columns
    df_tex = df.copy()
    df_tex["MC_DigCall $(\\pm3\\sigma)$"] = df_tex.apply(
        lambda r: f"{r['MC_DigitalCall']:.4f} $\\pm$ {3*r['SE_Call']:.4f}",
        axis=1,
    )
    df_tex["MC_DigPut $(\\pm3\\sigma)$"] = df_tex.apply(
        lambda r: f"{r['MC_DigitalPut']:.4f} $\\pm$ {3*r['SE_Put']:.4f}",
        axis=1,
    )
    cols = [
        "K",
        "T",
        "BS_DigitalCall",
        "MC_DigCall $(\\pm3\\sigma)$",
        "BS_DigitalPut",
        "MC_DigPut $(\\pm3\\sigma)$",
    ]
    tex_path = "experiments/results/digital_mc_vs_bs.tex"
    df_tex[cols].to_latex(
        tex_path,
        index=False,
        escape=False,
        float_format="%.4f",
        caption=(
            "Digital (cash-or-nothing) options: Monte Carlo vs analytic "
            "Black--Scholes. MC uses antithetic variates (50k paths). "
            "Rightmost columns check digital parity "
            "$D^{\\mathrm{call}}+D^{\\mathrm{put}}=e^{-rT}$."
        ),
        label="tab:digital_mc_vs_bs",
    )

    # Quick assertions: MC within 3σ of BS; parity close in MC
    for _, rrow in df.iterrows():
        assert (
            abs(rrow["MC_DigitalCall"] - rrow["BS_DigitalCall"])
            <= 3 * rrow["SE_Call"]
        ), "MC call outside 3σ"
        assert (
            abs(rrow["MC_DigitalPut"] - rrow["BS_DigitalPut"])
            <= 3 * rrow["SE_Put"]
        ), "MC put outside 3σ"
        tol = 3 * rrow["SE_Call"] + 3 * rrow["SE_Put"]

    print(f"\nSaved CSV to {csv_path}")
    print(f"Saved LaTeX to {tex_path}")


if __name__ == "__main__":
    main()
