"""
models/fx_sde.py

Hybrid Rough-Local-Stochastic Volatility FX simulator core module.
Contains FXHybridSimulator class for Garman-Kohlhagen spot path generation.
"""

import numpy as np
import matplotlib.pyplot as plt

class FXHybridSimulator:
    """
    Simulate FX spot S together with (optional) rough/stochastic variance on a shared grid.

    Args:
        S0:      Initial FX spot.
        sigma:   Constant spot volatility (used when v_model is None).
        T:       Time horizon (years).
        n_steps: Number of time steps per path.
        n_paths: Number of Monte Carlo paths.
        seed:    RNG seed.

    Notes:
        - If a variance model `v_model` is supplied, it must expose
          `generate_paths_from_increments(dW_v)` on the same grid (T, n_steps, n_paths).
        - Correlation across FX and rates is applied via a Cholesky factor `corr_L`
          in factor order (FX, rd, rf). Extra factors allowed (m >= 3).
    """

    def __init__(self, S0: float, sigma: float, T: float, n_steps: int, n_paths: int, seed:int|None = None) -> None:
        
        self.S0 = S0
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.seed = seed
        self.dt = T / float(n_steps)

        # Local random generator
        self.rng = np.random.default_rng(seed)


    def generate_paths_coupled_hw(
                        self,
                        # Hull–White (OU) params for domestic and foreign
                        r_dom0: float = 0.03,
                        kappa_dom: float = 0.40,
                        sigma_dom: float = 0.01, 
                        r_for0: float = 0.02, 
                        kappa_for: float = 0.50, 
                        sigma_for: float = 0.012,
                        # Correlations 
                        corr_L: np.ndarray = None,  # Cholesky for factors [FX, rd, rf, ...], shape (m,m), m>=3
                        rho_sv: float = -0.5,
                        # Volatility model (optional)
                        v_model = None,
                        store_Z_tensor: bool = True

            ) -> dict:
        
        """
        Evolve FX, r_d, r_f jointly on the same grid using *shared correlated shocks*.
        Factor order for corr_L must start with [FX, rd, rf]. Extra factors are allowed.
        - If v_model is provided, FX is coupled to variance via rho_sv (rough/stochastic vol).
        - If v_model is None, FX uses constant self.sigma (no variance path is built).

        Returns:
            dict with keys:
            "time", 
            "X" (log S), 
            "S", 
            "W" (FX Brownian path),
            "r_dom", 
            "r_for", 
            and optionally "Z_corr" (if store_Z_tensor).
        """

        # ---------- corr_L setup ----------
        if corr_L is None:
            m = 3
            corr_L = np.eye(m, dtype=float)  # default: independent factors
        else:
            if corr_L.ndim != 2 or corr_L.shape[0] != corr_L.shape[1]:
                raise ValueError("corr_L must be a square matrix.")
            m = int(corr_L.shape[0])
            if m < 3:
                # We can support m<3 by zeroing missing factors, but most use-cases need (FX, rd, rf)
                # We expect columns to be in (FX, rd, rf) order.
                raise ValueError("corr_L must be at least 3x3 in (FX, rd, rf) order.")
            

        
        # -------  FX driver with antithetic + moment matching -------

        half = (self.n_paths + 1) // 2    # Round up when number of paths is odd
        Z_half = self.rng.standard_normal(size=(half, self.n_steps))
        Z_fx_raw = np.vstack([ Z_half, -Z_half ])[: self.n_paths]   # trim to n_paths rows (n_paths, n_steps) 
        
        if self.n_paths > 1: 
            col_mean = Z_fx_raw.mean(axis=0, keepdims=True)
            col_std = Z_fx_raw.std(axis=0, keepdims=True)
            safe_std = np.where(col_std > 0, col_std, 1.0)
            Z_fx_raw = (Z_fx_raw- col_mean) / safe_std

        # Optional variance coupling (skip entirely if v_model is None)
        v = None
        if v_model is not None:
            Z_v_raw = self.rng.standard_normal(size=(self.n_paths, self.n_steps))
            
            if self.n_paths > 1:
                col_mean = Z_v_raw.mean(axis=0, keepdims=True)
                col_std  = Z_v_raw.std(axis=0, keepdims=True)
                safe_std = np.where(col_std > 0, col_std, 1.0)
                Z_v_raw = (Z_v_raw - col_mean) / safe_std

            # couple them with 2x2 Cholesky so Corr(Z_S, Z_v) = rho_sv
            #     [Z_S]   [ 1      0              ] [Z_fx_raw]
            #     [Z_v] = [ rho_sv  sqrt(1-rho^2) ] [Z_v_raw ]
            L2 = np.array([[1.0, 0.0], [rho_sv, np.sqrt(max(1.0 - rho_sv**2, 0.0))]], dtype=float)
            
            Z_pair = np.stack([Z_fx_raw, Z_v_raw], axis=2)     # (n_paths, n_steps, 2)
            Y_pair = Z_pair @ L2.T                             # apply 2x2 chol
            Z_S    = Y_pair[:, :, 0]                           # FX shock (standard normal)
            Z_v    = Y_pair[:, :, 1]                           # variance shock (standard normal)
           
           # Build variance path from dW_v
            dW_v = Z_v * np.sqrt(self.dt)

            # sanity-check same grid
            if (v_model.n_steps != self.n_steps) or (v_model.n_paths != self.n_paths) or (abs(v_model.T - self.T) > 1e-12):
                raise ValueError("v_model grid (T, n_steps, n_paths) must match FXSimulator.")
            v_res = v_model.generate_paths_from_increments(dW_v)
            v = v_res["v"]     # shape (n_paths, n_steps+1)
        else:
            # No variance model: just use the prepared FX normals
            Z_S = Z_fx_raw

        # ------ Pre-allocate state -------
        time = np.linspace(0.0, self.T, self.n_steps + 1)
        X = np.zeros((self.n_paths, self.n_steps + 1))
        W = np.zeros_like(X)
        S = np.zeros_like(X)
        r_dom = np.zeros((self.n_paths, self.n_steps + 1))
        r_for = np.zeros((self.n_paths, self.n_steps + 1))

        X[:, 0] = np.log(self.S0)
        r_dom[:, 0] = r_dom0
        r_for[:, 0] = r_for0

       # ---------- theta constant  ----------
        theta_d = np.full((self.n_paths, self.n_steps), r_dom0, dtype=float)
        theta_f = np.full((self.n_paths, self.n_steps), r_for0, dtype=float)

        store_Z = np.zeros((self.n_paths, self.n_steps, m)) if store_Z_tensor else None
        sqrt_dt = np.sqrt(self.dt)

        # ---------- Time stepping with shared correlated shocks ----------

        for i in range(self.n_steps):

            # Build uncorrelated shocks, pin factor 0 = FX shock
            Z_uncorr = self.rng.standard_normal(size=(self.n_paths, m))
            Z_uncorr[:, 0] = Z_S[:, i]
            
            # Correlate
            Z_corr = Z_uncorr @ corr_L.T
            if store_Z is not None:
                store_Z[:, i, :] = Z_corr

            # Brownian increments per factor  (ORDER: FX, rd, rf)
            dW_S_i  = np.sqrt(self.dt) * Z_corr[:, 0]    # FX
            dW_rd_i = np.sqrt(self.dt) * Z_corr[:, 1] if m >= 2 else 0.0  # domestic rate
            dW_rf_i = np.sqrt(self.dt) * Z_corr[:, 2] if m >= 3 else 0.0  # foreign rate


            # Hull–White (OU) updates using the SAME shocks
            r_dom[:, i+1] = r_dom[:, i] + kappa_dom * (theta_d[:, i] - r_dom[:, i]) * self.dt + sigma_dom * dW_rd_i
            r_for[:, i+1] = r_for[:, i] + kappa_for * (theta_f[:, i] - r_for[:, i]) * self.dt + sigma_for * dW_rf_i
            
            # Volatility (rough/stochastic if v provided; else constant sigma)
            if v is not None:
                vol = np.sqrt(np.maximum(v[:, i], 0.0))     # left-point vol
            else:
                vol = self.sigma                            # scalar

            # Garman–Kohlhagen in log form
            drift = (r_dom[:, i] - r_for[:, i] - 0.5 * (vol ** 2)) * self.dt
            X[:, i+1] = X[:, i] + drift + (vol * dW_S_i)

            # Track FX Brownian path (optional but useful for debugging)
            W[:, i+1] = W[:, i] + dW_S_i

        S = np.exp(X)

        # Store results
        paths = {"time": time, "W": W, "X": X,"S": S, "r_dom": r_dom, "r_for": r_for}
        if store_Z_tensor:
            paths["Z_corr"] = store_Z   # correlated normals per factor, per step
        self.paths = paths       # needed by plot_paths
        
        return paths

    def plot_paths(self, n_plot: int = 10) -> None:

        """
        Plot simulated FX spot paths.
        """

        if self.paths is None:
            raise ValueError("Call generate_paths_coupled_hw() first.")

        time = self.paths['time']
        S = self.paths['S']

        plt.figure(figsize=(10,5))
        for j in range(min(n_plot, self.n_paths)):
            plt.plot(time, S[j], lw=0.8, alpha=0.7)
        plt.title(rf"FX Spot Paths - pre-draw Z ($\sigma$={self.sigma})")
        plt.xlabel("Time (years)")
        plt.ylabel("Spot Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.show()