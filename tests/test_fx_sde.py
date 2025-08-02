# tests/test_fx_sde.py

import numpy as np
from models.fx_sde import FXSimulator

# Simulate 3-factor (FX, rd, rf) model in one call

def _simulate_fx(n_paths=10000, n_steps=126):

    """Return (paths, R_target)."""

    T, S_0, sigma = 1.0, 1.10, 0.15

    # flat-rate arrays so drift is deterministic
    rd = 0.03
    rf = 0.01
    rd_paths = np.full((n_paths, n_steps + 1), rd)
    rf_paths = np.full_like(rd_paths, rf)

    # target correlation matrix  (FX, rd, rf)
    R = np.array([[1.00, 0.30, -0.20],
                  [0.30, 1.00,  0.00],
                  [-0.20, 0.00, 1.00]])
    L = np.linalg.cholesky(R)

    fx = FXSimulator(S_0=S_0, sigma=sigma,
                     T=T, n_steps=n_steps, n_paths=n_paths, seed=42)

    paths = fx.generate_paths_with_rates(rd_paths, rf_paths, corr_L=L, store_Z_tensor=True)

    return paths, R


# Shape sanity

def test_shapes():
    paths, _ = _simulate_fx(500, 30)
    assert paths["S"].shape == (500, 31)
    assert paths["Z_corr"].shape[:2] == (500, 30)


# Empirical correlation ≈ target  (tolerance 3 %)

def test_correlation():
    paths, R_target = _simulate_fx(5000, 60)
    Z = paths["Z_corr"].reshape(-1, 3)          # collapse paths×steps
    R_emp = np.corrcoef(Z.T)
    assert np.max(np.abs(R_emp - R_target)) < 0.03


# Parity check

def test_put_call_parity():
   
   # tests/test_fx_sde.py  – replace test_put_call_parity
   T, n_steps = 1.0, 252
   paths, _   = _simulate_fx(n_paths=20_000, n_steps=n_steps)

   S0, rd, rf = 1.10, 0.03, 0.01
   S_T        = paths["S"][:, -1]
   D_d_T      = np.exp(-rd * T)

   # forward and ATM strike
   F_theo = S0 * np.exp((rd - rf) * T)
   K      = F_theo

   call = D_d_T * np.maximum(S_T - K, 0.0)
   put  = D_d_T * np.maximum(K - S_T, 0.0)
   raw  = call - put

   # control-variate with zero analytic mean
   cv    = D_d_T * (S_T - F_theo)
   beta  = np.cov(raw, cv, ddof=1)[0, 1] / np.var(cv, ddof=1)
   adj   = raw - beta * cv

   abs_err_bp = adj.mean() * 1e4        # bp of notional
   assert abs(abs_err_bp) < 1.0          # <= 1 bp
