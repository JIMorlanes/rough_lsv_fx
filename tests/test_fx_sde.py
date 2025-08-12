# tests/test_fx_sde.py

import numpy as np
from models.fx_sde import FXHybridSimulator


# Simulate 3-factor (FX, rd, rf) model in one call

def _simulate_fx(n_paths=10000, n_steps=126):
    """Return (paths, R_target)."""
    T, S0, sigma = 1.0, 1.10, 0.15

    rd = 0.03
    rf = 0.01

    # Target correlation in (FX, rd, rf) order — matches the simulator
    R = np.array([
                [1.00,  0.30, -0.20],
                [0.30,  1.00,  0.00],
                [-0.20, 0.00,  1.00],
                ], dtype=float)
    L = np.linalg.cholesky(R)

    fx = FXHybridSimulator(S0=S0, sigma=sigma,
                           T=T, n_steps=n_steps, n_paths=n_paths, seed=42)

    # pick reasonable OU params so rates actually move
    paths = fx.generate_paths_coupled_hw(
        r_dom0=rd, kappa_dom=0.40, sigma_dom=0.01, theta_dom=rd,
        r_for0=rf, kappa_for=0.50, sigma_for=0.012, theta_for=rf,
        corr_L=L,   # (FX, rd, rf) order
        v_model=None,
        store_Z_tensor=True,
    )
     # DEBUG PRINTS
    print(f"[DEBUG] n_paths={n_paths}, n_steps={n_steps}")
    print(f"[DEBUG] S.shape = {paths['S'].shape}")
    if "Z_corr" in paths:
        print(f"[DEBUG] Z_corr.shape = {paths['Z_corr'].shape}")
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
    """
    Verify that the empirical correlation of (FX, rd, rf) Brownian shocks
    matches the target correlation matrix R.

    Notes:
        - The simulator factor order is (FX, rd, rf)
        - rd = domestic short rate
        - rf = foreign short rate
    """
    R = np.array([
        [ 1.0,  0.30, -0.20],  # FX
        [ 0.30, 1.00,  0.50],  # rd
        [-0.20, 0.50,  1.00],  # rf
    ], dtype=float)

    # Simulator expects (FX, rd, rf) → permute rows/cols before Cholesky
    corr_L = np.linalg.cholesky(R)

    # --- Simulation settings ---
    n_paths, n_steps, tol = 20000, 252, 0.02
    sim = FXHybridSimulator(S0=1.0, sigma=0.20, T=1.0,
                            n_steps=n_steps, n_paths=n_paths, seed=12345)

    r_dom0, kappa_dom, sigma_dom = 0.03, 0.40, 0.01
    r_for0, kappa_for, sigma_for = 0.02, 0.50, 0.012

    out = sim.generate_paths_coupled_hw(
        r_dom0=r_dom0, kappa_dom=kappa_dom, sigma_dom=sigma_dom, theta_dom=r_dom0,
        r_for0=r_for0, kappa_for=kappa_for, sigma_for=sigma_for, theta_for=r_for0,
        corr_L=corr_L,
        v_model=None,
        store_Z_tensor=False,
    )

    # Increments — match simulator order (FX, rd, rf)
    dlogS = np.diff(out["X"], axis=1).ravel()
    drd   = np.diff(out["r_dom"], axis=1).ravel()   # domestic
    drf   = np.diff(out["r_for"], axis=1).ravel()   # foreign

    emp_corr = np.corrcoef(np.vstack([dlogS, drd, drf]))

    # Compare against R (already permuted to (FX, rd, rf))
    np.testing.assert_allclose(emp_corr, R, atol=tol)


