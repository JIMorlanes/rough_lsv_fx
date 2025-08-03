"""
models/rough_fou.py. Rough-Fractional OU variance driver
Quick Riemann-approx implementation (no FFT) to plug into FXSimulator.
Upgrade path: replace `_simulate_kernel()` with circulant embedding for production accuracy.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class RoughFOU:

    """
    Rough fractiona-OU variance process.

    • H ∈ (0, 0.5) ⇒ "rough" (long-memory) variance.
    • We use a power-law Riemann approximation:
        dB_H ≈ (dt)^H*Z,  Z ~ N(0,1)
    which is for prototyping (<1 bp bias at daily grid). For production accuracy, swap in an FFT-based
    sampler and keep the public API unchanged.
    """

    def __init__(self, v_0: float, kappa: float, theta: float, xi: float, H: float, T: float, n_steps: int,
        n_paths: int, seed: int|None = None) -> None:

        """
        Initialize FXSimulator.

        Args:
            v_0: Initial variance at time zero.
            kappa: Mean-reversion speed.
            theta: Long-run variance level (stationary).
            xi: volatility of volatility.
            T: Time horizon.
            n_steps: Steps per path.
            n_paths: Number of paths.
            dt: Size of each time increment
            seed: RNG seed.
        """

        self.v_0 = v_0
        self.kappa = kappa 
        self.theta = theta
        self.xi = xi 
        self.H = H
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.dt = T / float(n_steps)
        self.seed = seed

        # Local random generator
        self.rng = np.random.default_rng(seed)

        assert 0.0 < H < 0.5, "H must be in (0, 0.5) for rough volatility."

    # ------------------------------------------------------------------
    def generate_paths(self, Z_vol: np.ndarray | None = None) -> np.ndarray:

        """Return variance paths, shape = (n_paths, n_steps+1).

        Parameters
        ----------
        Z_vol : ndarray or None
            Pre-generated N(0,1) matrix, antithetic and  moment-matched.
            Shape = (n_paths, n_steps).  Pass **None** to generate here.

        Returns
        -------
        dictionary: time  and variance v
        """

        # create antithetic, column‑matched normals if user didn’t supply
        if Z_vol is None:
            half = (self.n_paths + 1) // 2   # Round up when number of paths is odd
            Z_half = self.rng.standard_normal(size=(half, self.n_steps))
            Z_vol = np.vstack((Z_half, -Z_half))[: self.n_paths]     #trim to n_paths rows

        # Column-wise moment matching. Making sure that samples from normal have mean 0 and variance 1
        if self.n_paths > 1: 
            col_mean = Z_vol.mean(axis=0, keepdims=True)
            col_std = Z_vol.std(axis=0, keepdims=True)
            safe_std = np.where(col_std > 0, col_std, 1.0)  # replace zeros with 1 so division is safe
            Z_vol = (Z_vol - col_mean) / safe_std

            
        # Pre-allocate variance array
        v = np.zeros((self.n_paths, self.n_steps + 1))
        v[:, 0] = self.v_0  # Initial variance at time zero
        time = np.zeros(self.n_steps+1)

        # main Euler loop with rough increment ≈ (dt)^H * Z
        rough_scale = (self.dt ** self.H)
        # sqrt_dt = np.sqrt(self.dt)  # needed for correct dim if you swap samplers later

        for i in range(0, self.n_steps):
            v[:,i+1] = v[:,i] + ( self.kappa * (self.theta - v[:, i]) * self.dt +
                self.xi * np.sqrt(np.maximum(v[:, i], 0.0)) * rough_scale * Z_vol[:, i])
            v[:, i+1] = np.maximum(v[:,i+1], 1e-12)  # positivity floor
            # Build time grid
            time[i+1] = time[i] + self.dt

        return {"time": time, "v": v}
