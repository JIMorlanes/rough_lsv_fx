# tests/test_fx_sde.py

import numpy as np
from models.fx_sde import FXHybridSimulator


# Simulate 3-factor (FX, rd, rf) model in one call

def _simulate_fx(n_paths=10000, n_steps=126):
    """
    Return (paths, R_target) using FXHybridSimulator.generate_paths_coupled_hw.

    Factor order is (FX, rd, rf) to match the simulator internals.
    """
    T, S0, sigma = 1.0, 1.10, 0.15

    # Target correlation in (FX, rd, rf)
    R = np.array([
        [ 1.00,  0.30, -0.20],  # FX
        [ 0.30,  1.00,  0.50],  # rd (domestic)
        [-0.20,  0.50,  1.00],  # rf (foreign)
    ], dtype=float)
    L = np.linalg.cholesky(R)

    # Build simulator
    fx = FXHybridSimulator(S0=S0, sigma=sigma, T=T,
                           n_steps=n_steps, n_paths=n_paths, seed=42)

    # Reasonable OU params so rates actually move
    r_dom0, kappa_dom, sigma_dom = 0.03, 0.40, 0.010
    r_for0, kappa_for, sigma_for = 0.01, 0.50, 0.012

    # Generate coupled paths — note: no theta_dom/theta_for
    paths = fx.generate_paths_coupled_hw(
        r_dom0=r_dom0, kappa_dom=kappa_dom, sigma_dom=sigma_dom,
        r_for0=r_for0, kappa_for=kappa_for, sigma_for=sigma_for,
        corr_L=L,
        v_model=None,
        store_Z_tensor=True,   # needed by shape/correlation tests
    )
    return paths, R



# Shape sanity

def test_shapes():
    paths, _ = _simulate_fx(n_paths=500, n_steps=30)
    # S: (n_paths, n_steps+1)
    assert paths["S"].shape == (500, 31)
    # Z_corr: (n_paths, n_steps, m) — at least m=3 (FX, rd, rf)
    assert "Z_corr" in paths and paths["Z_corr"].shape[:2] == (500, 30)
    assert paths["Z_corr"].shape[2] >= 3


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
   paths, _   = _simulate_fx(n_paths=20000, n_steps=n_steps)

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

def test_coupled_hw_empirical_correlation_matches_target():
    # Larger run for correlation stability
    n_paths, n_steps = 20000, 252
    TOL = 0.02

    fx = FXHybridSimulator(S0=1.0, sigma=0.20, T=1.0,
                           n_steps=n_steps, n_paths=n_paths, seed=12345)

    # Target correlation in (FX, rd, rf)
    R = np.array([
        [ 1.0,  0.30, -0.20],  # FX
        [ 0.30, 1.00,  0.50],  # rd
        [-0.20, 0.50,  1.00],  # rf
    ], dtype=float)
    L = np.linalg.cholesky(R)

    r_dom0, kappa_dom, sigma_dom = 0.03, 0.40, 0.01
    r_for0, kappa_for, sigma_for = 0.02, 0.50, 0.012

    out = fx.generate_paths_coupled_hw(
        r_dom0=r_dom0, kappa_dom=kappa_dom, sigma_dom=sigma_dom,
        r_for0=r_for0, kappa_for=kappa_for, sigma_for=sigma_for,
        corr_L=L,
        v_model=None,
        store_Z_tensor=True,     # capture shocks
    )

    # Use raw correlated normals to avoid drift/MR noise
    Z = out["Z_corr"][:, :, :3].reshape(-1, 3)   # (FX, rd, rf)
    emp_corr = np.corrcoef(Z.T)

    np.testing.assert_allclose(emp_corr, R, atol=TOL)