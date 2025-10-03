
# rough_lsv_fx

**Hybrid Rough / Local-Stochastic Volatility Engine for Multi-Currency FX Pricing**

This project models multi-currency FX dynamics using a combination of **rough volatility**, **local-stochastic volatility (LSV)**, and **interest rate term structure**. It is designed to reflect real-world pricing behavior seen in modern derivatives markets, particularly in the context of **XVA**, **structured FX**, and **front-office trading desks**.

The framework blends theoretical with practical implementation, drawing on:
- Fractional Brownian motion and fractional Ornstein–Uhlenbeck processes
- Garman–Kohlhagen-style FX dynamics extended with stochastic rates
- Monte Carlo simulation with optional variance reduction
- Smiles and skews fit via **eSSVI** surfaces; used to extract Dupire local vol for SLV

---

## Project Structure

```text
rough_lsv_fx/
│
├── docs/
│   ├── calibration.md         # Detailed breakdown of the eSSVI calibration process
├── datasets/                 # market vols, yield curves
│
├── models/
│   ├── fx_sde.py                 # Garman–Kohlhagen FX SDE with hybrid vol and stochastic rates
│   ├── hull_white.py             # Hull–White short-rate model
│   ├── local_vol.py              # Dupire local volatility surface generator
│   ├── rough_fou.py              # Rough fOU variance process (Volterra kernel)
│   └── rough_heston_volterra.py  # Rough Heston model via Volterra discretisation (production-ready)
├── pricing/
│   ├── mc_pricer.py          # Monte Carlo engine, variance reduction
│   └── greeks_adjoint.py     # pathwise and adjoint Greeks
├── calibration/
│   ├── eSSVI_fit.py          # surface calibration to vanilla FX quotes
│   └── local_svol_bridge.py  # map rough‑vol params ↔ market smile
├── notebooks/
│   ├── 01_paths.ipynb                       # Simulate basic FX paths under GH dynamics
│   ├── 02_rates_proto.ipynb                 # Prototyping Hull–White stochastic interest rates
│   ├── 03_fx_hw.ipynb                       # Combine FX and rate dynamics in full SDE
│   ├── 04_rough_fou.ipynb                   # Simulate fOU (rough volatility) process
│   ├── 05_rough_vol_reference.ipynb         # Reference rough volatility paths with H < 0.5
│   ├── 05a_rough_vol_step_by_step.ipynb     # Walkthrough of rough Heston simulation with plots
│   ├── 06_eSSVI_surface_prep.ipynb          # Convert market quotes to (T, k, sigma) format
│   ├── 06b_isotonic_regression.ipynb        # Enforce monotonic ATM vol term structure
│   ├── 07_fit_eSSVI_from_targets.ipynb      # Calibrate eSSVI surface to market points
│   └── 08_dupire_and_leverage.ipynb         # Extract Dupire local vol and compute SLV leverage
└── README.md
```

---

## Goals

- Build a modular, testable FX pricing library in Python
- Reproduce core components of front-office quant libraries
- Demonstrate technical depth in simulation, calibration, and risk analysis
- Serve as a showcase project for quantitative interviews

---

## Theory Background

- Garman–Kohlhagen FX pricing model
- Fractional processes (fOU, rough volatility, H < 0.5)
- SVI / eSSVI volatility modeling
- Hull–White short rate models
- Monte Carlo Greeks (adjoint, pathwise)

---

## Motivation

This project is inspired by real-world quant models used on **FX and XVA desks**. It reflects a blend of:
- Academic research in stochastic and fractional processes
- Practical needs for pricing, hedging, and managing exotic derivatives
- A desire to demonstrate readiness for **front-office quant roles** with production-quality modeling

---

## Status

- Garman–Kohlhagen simulation: completed
- FX simulator: completed
- Smile fitting and calibration: in progress
- Exotic pricing and hedge testing: upcoming

---

## Calibration Module

- `06_eSSVI_surface_prep.ipynb`:  
  Converts FX market quotes (ATM, 25Δ P/C) into `(T, k, sigma)` targets for surface fitting.  
  Includes a calendar monotonicity check on ATM variance.

- `07_fit_eSSVI_from_targets.ipynb`:  
  Will calibrate the eSSVI surface to market points using no-arbitrage constraints.

Documentation available in the [docs/](docs/) folder.

See full breakdown in [docs/calibration.md](docs/calibration.md)

---

## Requirements

Install from `requirements.txt` or use `environment.yml`.

```bash
pip install -r requirements.txt
```

or

```bash
conda env create -f environment.yml
```
