# Exotic Derivatives Pricing & Risk Library

Stack: Python (`NumPy`, `SciPy`); tests with `pytest`; viz with `Matplotlib`/`Streamlit`; Docs: Markdown.

<p align="center"> <b>Status:</b> Phase 1 in progress · <b>Target completion:</b> June 2026 </p>


## Overview


This project is an attempt to replicate a FO quantitative research environment from first principles. It is a compact quantitative pricing and hedging library for equity/FX options. At its core, the library is structured around:

* Model layer: stochastic dynamics (Black–Scholes, Heston, etc.)
* Numerical engines: Monte Carlo and PDE solvers
* Payoff layer: European, Asian, and barrier-style instruments
* Greeks & calibration layer: pathwise/likelihood estimators and fitting routines
* Validation layer: convergence, control variates, and analytical benchmarks

### Validation philosophy

Each model and numerical scheme is benchmarked against analytical references (when possible), stress-tested for convergence and bias, and documented.

A comprehensive technical report (report.pdf) in the repository root explains the models, numerical schemes, and validation experiments. All numerical experiments and validation plots used in the technical report are reproducible from the scripts in `experiments/`.



---

## Roadmap

This project is divided into 5 main phases. By the end of it, this repository will host a validated Python/C++ pricing library for vanilla and exotic derivatives, a Streamlit dashboard for interactive use, and a reproducible technical report explaining the implementation.

### Phase 1: Core pricers & greeks

- [X] Black-Scholes analytical pricers for vanilla, digital, and geometric Asian options.
- [X] Monte Carlo pricing engine with antithetic and control variates.
- [X] Pathwise and finite-difference estimators for $\Delta$.
- [ ] PDE solver (Crank-Nicolson) for barrier options.
- [X] Unit and convergence tests (Monte Carlo vs closed-form).
- [ ] Write corresponding report sections.

### Phase 2: Calibration & Model Fitting

- [ ] Implement Heston stochastic volatility model & build calibration engine.
- [ ] Fourier-Cosine (COS) method pricer for European options.
- [ ] Generate "market" vols from known Heston parameters, calibrate back, plot fit heatmap.
- [ ] Write corresponding report sections.

### Phase 3: Hedging Error Simulation

- [ ] Simulate discrete hedging under Black-Scholes and Heston dynamics.
- [ ] Compute hedging PnL distribution across rebalance frequencies (with transaction costs).
- [ ] Write corresponding report sections.


### Phase 4: Quanto Extension

- [ ] Extend library to handle foreign-underlying / domestic-currency payoffs.
- [ ] Implement Quanto drift adjustments.
- [ ] Write corresponding report sections.


### Phase 5: Dashboard & Final Report

- [ ] Build a Streamlit dashboard for interactive use.
- [ ] Finish the technical report + 1-page executive summary.

---

## Install & Run

### Installation using pip

```{bash}
pip install git+https://github.com/fjayousi/exotic-pricing-lib.git
```

### Run unit tests

```{bash}
pytest -v
```

---

## Deliverables

- Modular Python pricing library
- Validation notebooks and plots
- Unit and convergence test suite
- Technical report (in progress)
- Streamlit dashboard prototype (Phase 5)
