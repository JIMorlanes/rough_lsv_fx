# rough_lsv_fx

**Hybrid Rough / Local-Stochastic Volatility Engine for Multi-Currency FX Pricing**

This project models multi-currency FX dynamics using a combination of **rough volatility**, **local-stochastic volatility (LSV)**, and **interest rate term structure**. It is designed to reflect real-world pricing behavior seen in modern derivatives markets, particularly in the context of **XVA**, **structured FX**, and **front-office trading desks**.

The framework blends theoretical with practical implementation, drawing on:
- Fractional Brownian motion and fractional Ornstein–Uhlenbeck processes
- Garman–Kohlhagen-style FX dynamics extended with stochastic rates
- Monte Carlo simulation with optional variance reduction
- Smiles and skews fit via SVI, SABR, or local vol surfaces

---

## 🔧 Project Structure
```text
rough_lsv_fx/
├── data/                     # market vols, yield curves
├── models/
│   ├── hull_white.py         # stochastic short‑rate
│   ├── rough_fou.py          # fractional‑OU variance driver
│   ├── local_vol.py          # Dupire grid or SVI fit
│   └── fx_sde.py             # GK spot SDE with hybrid vol + rates
├── pricing/
│   ├── mc_pricer.py          # Monte Carlo engine, variance reduction
│   └── greeks_adjoint.py     # pathwise and adjoint Greeks
├── calibration/
│   ├── sabr_fit.py
│   └── local_svol_bridge.py  # map rough‑vol params ↔ market smile
├── notebooks/
│   ├── 01_paths.ipynb
│   ├── 02_smile_fit.ipynb
│   ├── 03_barrier_knockout.ipynb
│   └── 04_hedge_perf.ipynb
└── README.md
```
---

## ✨ Goals

- Build a modular, testable FX pricing library in Python
- Reproduce core components of front-office quant libraries
- Demonstrate technical depth in simulation, calibration, and risk analysis
- Serve as a showcase project for quantitative interviews

---

## 📚 Theory Background

- Garman–Kohlhagen FX pricing model
- Fractional processes (fOU, rough volatility, H < 0.5)
- SABR / SVI volatility modeling
- Hull–White short rate models
- Monte Carlo Greeks (adjoint, pathwise)

---

## 🧠 Motivation

This project is inspired by real-world quant models used on **FX and XVA desks**. It reflects a blend of:
- Academic research in stochastic and fractional processes
- Practical needs for pricing, hedging, and managing exotic derivatives
- A desire to demonstrate readiness for **front-office quant roles** with production-quality modeling

---

## 🗂️ Status

✅ Initial Garman–Kohlhagen simulation  
⏳ Refactor to `FXSimulator` class  
🔜 Smile fitting and stochastic volatility  
🔜 Barrier option pricing and hedge testing


## 📎 Requirements

Install from `requirements.txt` or activate `environment.yml`.

pip install -r requirements.txt

or use conda

conda env create -f environment.yml

