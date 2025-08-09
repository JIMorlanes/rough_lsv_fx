"""
models/rough_heston_volterra.py

Rough Heston variance process via Volterra discretisation.

Continuous-time:
    v(t_i) = v0
           + integral{0 to t_i} K(t_i - s) * kappa * (theta - v(s)) ds
           + integral{0 to t_i} K(t_i - s) * xi * sqrt(v(s)) dW_s

with K(t) = t^(H - 1/2) / Gamma(H + 1/2),  0 < H < 1/2.

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

        # Local random generator
        self.rng = np.random.default_rng(seed)

        # Time grid
        self.dt = self.T / self.n_steps
        self.time = np.linspace(0.0, self.T, self.n_steps + 1)

        # Precompute Volterra weights
        self.w_dt, self.w_dW = self._precompute_weights(self.n_steps, self.dt, self.H)

    # -------------------------------
    # Internal helpers 
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
        
        Returns:
            w_dt (ndarray): shape (n_steps,), exact integral of K over each cell [(n-1)dt, n dt].
            w_dW (ndarray): shape (n_steps,), K evaluated at midpoints (n - 1/2) dt.
        """
        n = np.arange(1, n_steps + 1, dtype=float)

        # Stochastic (midpoints)
        t_mid = (n - 0.5) * dt
        w_dW_mid = self._k_power(t_mid, H)

        # Deterministic (exact integral over each cell)
        Hp = H + 0.5
        t_right = n * dt
        t_left = (n - 1.0) * dt
        w_dt_exact = ( (t_right ** Hp) - (t_left ** Hp) ) / (math.gamma(H + 0.5) * Hp)

        return w_dt_exact, w_dW_mid

    def _draw_increments(self):

        """
        Brownian increments with antithetic pairing + moment matching.
        Returns Z * sqrt(dt) of shape (n_paths, n_steps).
        """

        # create antithetic, column‑matched normals
        half = (self.n_paths + 1) // 2   # Round up when number of paths is odd
        Z_half = self.rng.standard_normal(size=(half, self.n_steps))
        Z = np.vstack((Z_half, -Z_half))[: self.n_paths]     #trim to n_paths rows

       # Column-wise moment matching. Making sure that samples from normal have mean 0 and variance 1
        if self.n_paths > 1: 
            col_mean = Z.mean(axis=0, keepdims=True)
            col_std = Z.std(axis=0, keepdims=True)
            safe_std = np.where(col_std > 0, col_std, 1.0)  # replace zeros with 1 so division is safe
            Z = (Z - col_mean) / safe_std

        return Z * math.sqrt(self.dt)

    # -------------------------------
    # Public API
    # -------------------------------
    
    def generate_paths(self):
        """
        Direct Volterra simulation (O(n_steps^2) per path). Returns t (grid) and v (paths).

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
    
    # -------------------------------
    # Plotting helpers 
    # -------------------------------

    def plot_kernel(self):
        """
        Plot fractional kernel K(t) on [dt/2, T]
        """
        import matplotlib.pyplot as plt

        t = np.linspace(self.dt / 2, self.T, 400)
        Kt = self._k_power(t, self.H)

        plt.figure(figsize=(7, 3))
        plt.plot(t, Kt)
        plt.title("Fractional kernel K(t)")
        plt.xlabel("t")
        plt.ylabel("K(t)")
        plt.grid(True)
        plt.show()

    def plot_midpoint_weights(self):
        """
        Plot stochastic midpoint weights w_dW vs time midpoints (plain LaTeX for mathtext).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        n = np.arange(1, len(self.w_dW) + 1, dtype=float)
        t_mid = (n - 0.5) * self.dt

        plt.figure(figsize=(7, 3))
        plt.plot(t_mid, self.w_dW, marker=".", linestyle="none",label=r"$w_{dW}$ (stochastic midpoint)")
        plt.title("Stochastic midpoint weights $w_{dW}$ from model")
        plt.xlabel("time midpoint (t)")
        plt.ylabel(r"$w_{dW}$")
        plt.grid(True)
        plt.show()


    def plot_cell_weights(self):
        """
        Plot deterministic cell-integrated weights w_dt vs cell end times (plain LaTeX).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        n = np.arange(1, len(self.w_dt) + 1, dtype=float)
        t_right = n * self.dt

        plt.figure(figsize=(7, 3))
        plt.plot(t_right, self.w_dt, marker=".", linestyle="none")
        plt.title(r"Cell-integrated weights $w_{\Delta t}$ (per cell)")
        plt.xlabel("cell end time (t)")
        plt.ylabel(r"$w_{\Delta t}$")
        plt.grid(True)
        plt.show()


    def plot_kernel(self):
        """
        Plot fractional kernel K(t) on [dt/2, T] (plain LaTeX).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        t = np.linspace(self.dt / 2, self.T, 400)
        Kt = self._k_power(t, self.H)

        plt.figure(figsize=(7, 3))
        plt.plot(t, Kt)
        plt.title(r"Fractional kernel $K(t)$")
        plt.xlabel("t")
        plt.ylabel(r"$K(t)$")
        plt.grid(True)
        plt.show()


    def plot_kernel_loglog(self):
        """
        Log-log K(t) to show slope approximately H - 1/2 (plain LaTeX).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        t = np.linspace(self.dt / 2, self.T, 400)
        Kt = self._k_power(t, self.H)

        plt.figure(figsize=(7, 3))
        plt.loglog(t, Kt)
        plt.title("K(t) on log-log scale (slope ≈ H - 1/2)")
        plt.xlabel("t (log)")
        plt.ylabel(r"$K(t)$ (log)")
        plt.grid(True, which="both")
        plt.show()

    def plot_paths(self, n_paths_to_show: int = 10, res: dict | None = None):
        """
        Plot a subset of variance paths (05a style).
        """
        import matplotlib.pyplot as plt

        if res is None:
            res = self.generate_paths()

        v = res["v"]
        t = res["time"]

        k = min(n_paths_to_show, v.shape[0])
        plt.figure(figsize=(9, 3))
        for i in range(k):
            plt.plot(t, v[i], lw=0.8, alpha=0.9)
        plt.title("Rough Heston variance paths")
        plt.xlabel("time")
        plt.ylabel("v(t)")
        plt.grid(True)
        plt.show()
    
   