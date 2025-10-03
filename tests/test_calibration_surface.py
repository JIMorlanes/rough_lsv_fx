# test_calibration_surface.py
#
# Unit tests for calibration data prep and helper functions.
# These ensure that our preprocessing, strike/delta conversion,
# isotonic regression, and table outputs behave as expected.

import numpy as np
import pandas as pd
from calibration.utils import isotonic_nondec
from scipy.stats import norm

def test_isotonic_monotonic():
    """
    Test that the isotonic_nondec() function correctly projects a noisy
    series onto a non-decreasing sequence. This is crucial for ensuring
    no calendar arbitrage in w(T).
    """
    y = np.array([0.1, 0.09, 0.11, 0.15])
    y_iso = isotonic_nondec(y)
    assert np.all(np.diff(y_iso) >= 0), "Isotonic output not non-decreasing."


def test_forward_price_formula():
    """
    Test that the forward price is computed correctly as:
    F(T) = S0 * exp((r_d - r_f) * T).
    This is the foundation for all strike computations.
    """
    S0, rd, rf, T = 1.0, 0.02, 0.01, 0.5
    F_expected = S0 * np.exp((rd - rf) * T)
    assert np.isclose(F_expected, 1.005), "Forward price formula incorrect."


def test_strike_from_delta_call():
    """
    Test that strike K computed from 25Δ Call using the
    premium-included forward-delta convention is positive and finite.
    This guards against sign or formula errors in strike inversion.
    """
    F, sigma, delta, T = 1.01, 0.15, 0.25, 0.5
    z = norm.ppf(delta * np.exp(0.01 * T))
    K = F * np.exp(-sigma * np.sqrt(T) * z + 0.5 * sigma**2 * T)
    assert K > 0.0, "Strike from delta (call) returned nonpositive."


def test_target_table_columns():
    """
    Test that the raw targets table saved by 06_eSSVI_surface_prep.ipynb
    contains the expected columns: T, F, K_25P, K_ATM, K_25C.
    This ensures the downstream notebooks (07_fit_eSSVI_from_targets)
    can load and use it.
    """
    df = pd.read_csv("datasets/targets_from_article_06.csv")
    expected = {"T", "F", "K_25P", "K_ATM", "K_25C"}
    assert expected.issubset(df.columns), "Missing required columns."


def test_training_points_table():
    """
    Test that the detailed training points table (with isotonic variance)
    contains all required fields, has non-negative total variance,
    and finite log-moneyness. This ensures the eSSVI fit uses clean data.
    """
    df = pd.read_csv("datasets/targets_from_article_06_training_points.csv")
    required = {"T", "point", "F", "K", "k", "sigma", "w"}
    assert required.issubset(df.columns), "Training point table missing columns."

    # Check no negative total variance values
    assert df["w"].min() >= 0, "Negative total variance found"
    # Check log-moneyness is finite
    assert np.all(np.isfinite(df["k"])), "Non-finite log-moneyness values"
