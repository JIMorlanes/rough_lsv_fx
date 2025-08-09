
import numpy as np
from models.rough_heston_volterra import RoughHestonVolterra


def test_floor_mode_nonnegative_and_respects_eps():
    """
    Test that when `reflect=False`, the variance process is floored at eps_floor.

    This ensures:
    1. No negative variance values appear in the simulation.
    2. The minimum variance is exactly (or above) the specified `eps_floor` value.
    """
    m = RoughHestonVolterra(
        v0=0.01, kappa=1.0, theta=0.02, xi=0.5, H=0.1,
        T=0.2, n_steps=32, n_paths=300, seed=1,
        reflect=False, eps_floor=1e-8
    )
    # Simulate variance paths
    v = m.generate_paths()["v"]
    # Assertion:
    # Every single simulated variance value must be >= eps_floor.
    # This means the flooring worked and no value fell below the allowed minimum.
    assert (v >= 1e-8).all()


def test_kernel_weights_strictly_decreasing():
    """
    Test that Volterra kernel weights are strictly monotonic non-increasing weights 
    with a tiny tolerance for numerical noise.

    For a fractional kernel K(t) = t^(H - 1/2) / Gamma(H + 1/2),
    the sequence of midpoint weights w_dW should decrease strictly
    as t increases, provided H < 0.5.
    """
    tol = 1e-14
    m = RoughHestonVolterra(
        v0=0.04, kappa=1.0, theta=0.05, xi=0.2, H=0.1,
        T=1.0, n_steps=128, n_paths=10, seed=0
    )

    # Calculate differences
    diffs = np.diff(m.w_dW)

    # Check: all differences should be negative within tolerance
    assert np.all(diffs < tol), (
        f"Kernel weights are not strictly decreasing. "
        f"Found non-decreasing diffs: {diffs[diffs >= tol]}"
    )


def test_mean_tends_toward_theta_rough_check():
    """
    Compares the empirical mean variance at the end of the simulation to the 
    theoritical long-run variance theta. the rough nature of the kernel and the 
    stochastic term add noise, so it does not require perfect equality.
    """
    m = RoughHestonVolterra(
        v0=0.10, kappa=2.0, theta=0.05, xi=0.10, H=0.1,
        T=1.0, n_steps=256, n_paths=5000, seed=3
    )
    v = m.generate_paths()["v"]
    mean_end = float(v[:, -1].mean())
    # Should be reasonably close to theta (rough model adds noise; use loose tol)
    assert abs(mean_end - m.theta) < 0.02


def test_mean_reversion_matches_theoretical():
    """
    Check that the empirical mean of v(t) matches the theoretical
    mean-reversion curve of the drift term (OU-like behavior) within tolerance.
    """
    m = RoughHestonVolterra(
        v0=0.04, kappa=1.0, theta=0.05, xi=0.01, H=0.1,
        T=1.0, n_steps=200, n_paths=5000, seed=42
    )

    res = m.generate_paths()
    v = res["v"]
    t = res["time"]

    mean_emp = v.mean(axis=0)
    mean_theo = m.theta + (m.v0 - m.theta) * np.exp(-m.kappa * t)

    # Tolerance: allow small stochastic bias
    max_abs_error = np.max(np.abs(mean_emp - mean_theo))
    assert max_abs_error < 5e-3, f"Mean reversion mismatch too large: {max_abs_error:.3e}"
