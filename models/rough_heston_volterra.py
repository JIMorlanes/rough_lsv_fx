"""
models/05_rough_heston_volterra.py

Rough Heston variance process via Volterra discretisation.

Continuous-time:
    v(t_i) = v0
           + integral{0^{t_i} K(t_i - s) * kappa * (theta - v(s)) ds}
           + integral{0^{t_i} K(t_i - s) * xi * sqrt(v(s)) dW_s},

with K(t) = t^(H - 1/2) / Γ(H + 1/2),  0 < H < 1/2.

Deterministic integral - exact per-cell weights.
Stochastic integral - midpoint kernel weights.
"""

import math
import numpy as np


class RoughHestonVolterra:
    """
    Rough Heston variance simulator (Volterra discretisation).
    
    Attributes:
        v0 (float): Initial variance level.
        kappa (float): Mean-reversion speed.
        theta (float): Long-run variance.
        xi (float): Vol-of-vol parameter.
        H (float): Hurst exponent, 0 < H < 0.5.
        T (float): Time horizon in years.
        n_steps (int): Number of time steps.
        n_paths (int): Number of Monte Carlo paths.
        seed (int | None): RNG seed.
        reflect (bool): Reflect negative variance if True, else floor at eps_floor.
        eps_floor (float): Minimum allowed variance when reflect=False.

    Methods:
        generate_paths(): Run the simulation and return variance paths and time grid.
    """

    def __init__(self,
                 v0: float,
                 kappa: float,
                 theta: float,
                 xi: float,
                 H: float,
                 T: float,
                 n_steps: int,
                 n_paths: int,
                 seed: int | None = None,
                 reflect: bool = True,
                 eps_floor: float = 1e-12) -> None:
        
        # Parameters (constant for this version)
        self.v0 = float(v0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.H = float(H)
        self.T = float(T)
        self.n_steps = int(n_steps)
        self.n_paths = int(n_paths)
        self.seed = seed
        self.reflect = bool(reflect)
        self.eps_floor = float(eps_floor)

        # Time grid
        self.dt = self.T / self.n_steps
        self.time = np.linspace(0.0, self.T, self.n_steps + 1)

        # Precompute Volterra weights on this grid
        self.w_dt, self.w_dW = self._precompute_weights(self.n_steps, self.dt, self.H)

    # -------------------------------
    # Private helpers 
    # -------------------------------

    def _k_power(self, t, H):
        """
        K(t) = t^(H-1/2) / Gamma(H+1/2).  t can be scalar or array.
        """
        t = np.asarray(t, dtype=float)
        return t ** (H - 0.5) / math.gamma(H + 0.5)

    def _precompute_weights(self, n_steps: int, dt: float, H: float):
        """
        Deterministic exact-cell weights and stochastic midpoint weights.
        """
        n = np.arange(1, n_steps + 1, dtype=float)

        # Stochastic (midpoints)
        t_mid = (n - 0.5) * dt
        w_dW_mid = self._k_power(t_mid, H)

        # Deterministic (exact integral over each cell)
        Hp = H + 0.5
        t_right = n * dt
        t_left = (n - 1.0) * dt
        w_dt_exact = (t_right ** Hp - t_left ** Hp) / (math.gamma(H + 0.5) * Hp)

        return w_dt_exact, w_dW_mid

    def _draw_increments(self):

        """
        Brownian increments with antithetic pairing + moment matching.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        half = self.n_paths // 2
        Z = np.random.randn(half, self.n_steps)
        Z = np.vstack([Z, -Z])  # antithetic

        if Z.shape[0] < self.n_paths:  # handle odd path counts
            extra = np.random.randn(1, self.n_steps)
            Z = np.vstack([Z, -extra])

        # Column-wise zero mean, unit std
        Z -= Z.mean(axis=0, keepdims=True)
        std = Z.std(axis=0, keepdims=True)
        std[std == 0.0] = 1.0
        Z /= std

        return Z * math.sqrt(self.dt)

    # -------------------------------
    # Public API
    # -------------------------------
    
    def generate_paths(self):
        """
        Simulate variance paths using the notebook-05 Volterra scheme.

        Returns:
            dict: {"v": (n_paths, n_steps+1) variance paths, "time": (n_steps+1,) grid}
        """
        v = np.zeros((self.n_paths, self.n_steps + 1), dtype=float)
        v[:, 0] = self.v0

        dW = self._draw_increments()  # (n_paths, n_steps)

        for i in range(1, self.n_steps + 1):
            # Past values up to i-1
            vi = v[:, :i]                 # shape (n_paths, i), i.e. [v_0, v_1, ..., v_{i-1}] per path
            Kj_dt = self.w_dt[:i][::-1]   # shape (i,), deterministic weights for lags i-1..0 (exact)
            Kj_dW = self.w_dW[:i][::-1]   # shape (i,), stochastic weights for lags i-1..0 (midpoint)

            # Deterministic Volterra term: sum_j K_{i-j} * kappa*(theta - v_j)
            # Note: w_dt already contains the dt integration.
            phi = self.kappa * (self.theta - vi)          # (n_paths, i)
            det = phi @ Kj_dt                             # convolution (n_paths,)

            # Stochastic Volterra term: sum_j K_{i-j} * xi*sqrt(v_j) * dW_j
            psi_dW = self.xi * np.sqrt(np.clip(vi, 0.0, None)) * dW[:, :i]
            sto = psi_dW @ Kj_dW

            # Update variance
            v[:, i] = self.v0 + det + sto

            # Positivity control (reflection or floor)
            if self.reflect:
                v[:, i] = np.abs(v[:, i])
            else:
                v[:, i] = np.maximum(v[:, i], self.eps_floor)

        return {"v": v, "time": self.time}
