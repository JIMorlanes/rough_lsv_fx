import numpy as np
from models.hull_white import HullWhiteModel

def test_hullwhite_basic():

    # parameters 
    r_0 = 0.03
    kappa = 0.5
    theta = 0.04
    eta = 0.10
    T = 1.0
    n_steps = 20
    n_paths = 10000   # small but enough for the mean check

    hw = HullWhiteModel(r_0=r_0, lambd=kappa, theta=theta, eta=eta, T=T, n_steps=n_steps,
        n_paths=n_paths, seed=123)
    res = hw.generate_paths()
    r, time = res["r"], res["time"]

    # shape sanity 
    assert r.shape == (n_paths, n_steps + 1)
    assert time.shape == (n_steps + 1,)

    # simple mean-reversion check at t = T 
    expected_mean_T = r_0 * np.exp(-kappa * T) + theta * (1 - np.exp(-kappa * T))
    empirical_mean_T = r[:, -1].mean()

    error_bp = (empirical_mean_T - expected_mean_T) / expected_mean_T * 10000  # basis points
    assert abs(error_bp) < 50, f"Mean error too large: {error_bp:.1f} bp (allowed 50 bp)"
