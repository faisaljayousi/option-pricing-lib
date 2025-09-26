"""
BS vs MC sweep for European calls/puts 

Generates:
  - CSV with full numeric results (+ run metadata header row)
  - LaTeX table with ± 3σ error bars

Usage:
  PYTHONPATH=src python experiments/bs_vs_mc.py --paths 50000 --steps-per-year 252 --outdir experiments/results
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from pricers.bs_vanilla import call_price, put_price
from pricers.mc_vanilla import price_european_vanilla_mc


# ------------------------------ Config ----------------------------------------
@dataclass(frozen=True)
class SweepConfig:
    S0: float = 100.0
    r: float = 0.02
    q: float = 0.01
    sigma: float = 0.20
    maturities: Tuple[float, ...] = (0.25, 0.50, 1.00, 2.00)
    strikes: Tuple[float, ...] = (80, 90, 100, 110, 120)

    # MC params
    paths_per_case: int = 50_000
    steps_per_year: int = 252
    antithetic: bool = True
    seed_base: int = 2_025_09_26  # any fixed integer

    # IO
    outdir: Path = Path("experiments") / "results"
    csv_name: str = "vanilla_bs_mc.csv"
    tex_name: str = "vanilla_bs_mc_table.tex"

    # Presentation
    ci_mult: float = 3.0  # show ± 3σ


# ------------------------------ Helpers ---------------------------------------
def case_seed(base: int, K: float, T: float) -> int:
    """
    Derive a deterministic seed per (K, T) from a base seed, stable across runs.
    Avoids collisions and keeps reproducibility when the grid changes.
    """
    # Simple integer hash (no external deps). We quantise to 1e-6 to be stable in str.
    key = f"{base}|K={K:.6f}|T={T:.6f}"
    # Fowler–Noll–Vo (FNV-1a) 32-bit
    h = 2166136261
    for ch in key.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    # Mix with base and keep within 32-bit signed range
    return int((h ^ base) & 0x7FFFFFFF)


def fmt_pm(mean: float, se: float, mult: float) -> str:
    return f"{mean:,.4f} ± {mult*se:,.4f}"


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def metadata_row(cfg: SweepConfig) -> Dict[str, Any]:
    """One tidy row of run metadata to prepend to CSV (makes provenance obvious)."""
    ts = datetime.now(timezone.utc).isoformat()
    commit = os.environ.get("GIT_COMMIT", "").strip()
    return {
        "meta": "run_metadata",
        "timestamp_utc": ts,
        "git_commit": commit,
        **{f"cfg_{k}": v for k, v in asdict(cfg).items() if k not in {"outdir"}},
        # keep paths separate
        "outdir": str(cfg.outdir),
    }


# ------------------------------ Core logic ------------------------------------
def mc_price_call_put(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps_per_year: int,
    paths: int,
    antithetic: bool,
    seed: int,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    steps = max(1, int(math.ceil(T * steps_per_year)))

    c_mc, c_se = price_european_vanilla_mc(
        S0,
        K,
        r,
        q,
        sigma,
        T,
        steps=steps,
        n_paths=paths,
        call=True,
        antithetic=antithetic,
        seed=seed,
    )

    # nudge seed for put so the two estimates are independent but reproducible
    p_mc, p_se = price_european_vanilla_mc(
        S0,
        K,
        r,
        q,
        sigma,
        T,
        steps=steps,
        n_paths=paths,
        call=False,
        antithetic=antithetic,
        seed=seed + 7,
    )
    return (c_mc, c_se), (p_mc, p_se)


def sweep(cfg: SweepConfig) -> pd.DataFrame:
    rows = []
    for T in cfg.maturities:
        steps = int(math.ceil(T * cfg.steps_per_year))
        for K in cfg.strikes:
            # analytic BS
            c_bs = call_price(cfg.S0, K, cfg.r, cfg.q, cfg.sigma, T)
            p_bs = put_price(cfg.S0, K, cfg.r, cfg.q, cfg.sigma, T)

            # MC
            seed = case_seed(cfg.seed_base, K, T)
            (c_mc, c_se), (p_mc, p_se) = mc_price_call_put(
                cfg.S0,
                K,
                cfg.r,
                cfg.q,
                cfg.sigma,
                T,
                steps_per_year=cfg.steps_per_year,
                paths=cfg.paths_per_case,
                antithetic=cfg.antithetic,
                seed=seed,
            )

            # errors / checks
            rows.append(
                {
                    "S0": cfg.S0,
                    "K": float(K),
                    "T": float(T),
                    "r": cfg.r,
                    "q": cfg.q,
                    "sigma": cfg.sigma,
                    "BS_call": c_bs,
                    "MC_call": c_mc,
                    "SE_call": c_se,
                    "abs_err_call": abs(c_mc - c_bs),
                    "within_3se_call": abs(c_mc - c_bs) <= cfg.ci_mult * c_se,
                    "BS_put": p_bs,
                    "MC_put": p_mc,
                    "SE_put": p_se,
                    "abs_err_put": abs(p_mc - p_bs),
                    "within_3se_put": abs(p_mc - p_bs) <= cfg.ci_mult * p_se,
                    "paths": cfg.paths_per_case,
                    "steps": steps,
                    "antithetic": cfg.antithetic,
                    "seed_case": seed,
                }
            )
    return pd.DataFrame(rows)


def to_compact_table(df: pd.DataFrame, ci_mult: float) -> pd.DataFrame:
    def row_fmt(rec) -> pd.Series:
        return pd.Series(
            {
                "K": int(rec["K"]),
                "T": f'{rec["T"]:.2f}',
                "BS Call": f'{rec["BS_call"]:.3f}',
                "MC Call ($\\pm {m}\\sigma$)".format(m=int(ci_mult)): fmt_pm(
                    rec["MC_call"], rec["SE_call"], ci_mult
                ),
                "BS Put": f'{rec["BS_put"]:.3f}',
                "MC Put ($\\pm {m}\\sigma$)".format(m=int(ci_mult)): fmt_pm(
                    rec["MC_put"], rec["SE_put"], ci_mult
                ),
            }
        )

    table = df.apply(row_fmt, axis=1)
    table = table.astype({"K": int})
    return table.sort_values(by=["T", "K"]).reset_index(drop=True)


def write_outputs(
    df: pd.DataFrame, table: pd.DataFrame, cfg: SweepConfig
) -> Tuple[Path, Path]:
    ensure_outdir(cfg.outdir)

    csv_path = cfg.outdir / cfg.csv_name
    tex_path = cfg.outdir / cfg.tex_name

    # CSV: prepend one metadata row (wide) then data rows.
    meta = pd.DataFrame([metadata_row(cfg)])
    out = pd.concat([meta, df], ignore_index=True)
    out.to_csv(csv_path, index=False)

    # LaTeX: compact, with caption/label and proper alignment
    latex_str = table.to_latex(
        index=False,
        escape=False,
        column_format="rrrrrr",
        caption=(
            "European option prices: Black--Scholes (analytic) vs Monte Carlo "
            "(with $\\pm {m}\\sigma$ error bars)."
        ).format(m=int(cfg.ci_mult)),
        label="tab:bs_vs_mc_vanillas",
    )
    tex_path.write_text(latex_str, encoding="utf-8")

    return csv_path, tex_path


# ------------------------------ CLI -------------------------------------------
def parse_args(argv: Iterable[str]) -> SweepConfig:
    p = argparse.ArgumentParser(
        description="BS vs MC sweep for European options (reproducible)."
    )
    p.add_argument("--paths", type=int, default=50_000, help="MC paths per case")
    p.add_argument(
        "--steps-per-year", type=int, default=252, help="Time steps per year"
    )
    p.add_argument(
        "--no-antithetic", action="store_true", help="Disable antithetic variates"
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=2_025_09_24,
        help="Base seed for case derivation",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments") / "results",
        help="Output directory",
    )
    p.add_argument(
        "--ci-mult",
        type=float,
        default=3.0,
        help="Error bar multiplier (e.g., 1.96 or 3.0)",
    )

    args = p.parse_args(list(argv))

    return SweepConfig(
        paths_per_case=args.paths,
        steps_per_year=args.steps_per_year,
        antithetic=not args.no_antithetic,
        seed_base=args.seed_base,
        outdir=args.outdir,
        ci_mult=args.ci_mult,
    )


def main(argv: Iterable[str]) -> int:
    cfg = parse_args(argv)
    print("[INFO] Running sweep with config:", cfg)

    df = sweep(cfg)
    table = to_compact_table(df, cfg.ci_mult)
    csv_path, tex_path = write_outputs(df, table, cfg)

    # Console preview (first few rows)
    print(f"[OK] Wrote CSV: {csv_path}")
    print(f"[OK] Wrote LaTeX table: {tex_path}")
    print("\nPreview:")
    with pd.option_context("display.max_rows", 12, "display.width", 120):
        print(table.head(8).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

