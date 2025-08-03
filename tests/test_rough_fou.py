
import numpy as np
from models.rough_fou import RoughFOU

# This test is checking that the simulated variance process exhibits the correct mean-reversion behavior
def test_roughfou_basic_mean_reversion():
    fou = RoughFOU(v_0=0.04, kappa=1.0, theta=0.05, xi=0.01, H=0.1, T=1.0, n_steps=100, n_paths=5000, seed=42)
    res = fou.generate_paths()
    v = res["v"]
    t = res["time"][50]
    expected = 0.05 + (0.04 - 0.05) * np.exp(-1.0 * t)
    empirical = v[:, 50].mean()
    
    error_bp = (empirical - expected) / expected * 10000  # basis points
    # allow, say, 50 bp deviation (~0.5%)
    assert abs(error_bp) < 50, f"Mean reversion error too large: {error_bp:.1f} bp"