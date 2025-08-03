"""
models/fx_sde.py

Hybrid Rough-Local-Stochastic Volatility FX simulator core module.
Contains FXSimulator class for Garman-Kohlhagen spot path generation.
"""

import numpy as np
import matplotlib.pyplot as plt

class FXSimulator:

    """
    FX spot rate simulator using the Garman-Kohlhagen model.
    Supports constant or time-dependent domestic/foreign rates,
    pre-draws the full normal matrix for speed, and retains
    the full Wiener and time grids for later analytics.
    """

    def __init__(self, S_0: float, sigma: float, T: float, n_steps: int, n_paths: int, seed:int|None = None) -> None:

        """
        Initialize FXSimulator.

        Args:
            S_0: Initial FX spot.
            sigma: Spot volatility.
            T: Time horizon.
            n_steps: Steps per path.
            n_paths: Number of paths.
            seed: RNG seed.
        """

        self.S_0 = S_0
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.seed = seed
        self.dt = T / float(n_steps)

        # Local random generator
        self.rng = np.random.default_rng(seed)


    def generate_paths_with_rates(self, r_dom_paths: np.ndarray|None=None, r_for_paths:np.ndarray|None=None,
                                  corr_L: np.ndarray|None=None, r_dom: float|None=None, r_for: float|None=None,
                                  store_Z_tensor: bool = True) -> dict:
        
        """
        Simulate FX spot paths, storing full time grid and Wiener process.

        Parameters:
            r_dom_paths, r_for_paths
                Arrays of shape *(n_paths, n_steps+1)* with pre-simulated
                Hull-White short-rate paths.  If *None*, provide scalar
                `r_dom`, `r_for` instead.
            corr_L
                Lower-triangular Cholesky factor (m x m).  First column must
                correspond to the FX shock.  `None` ⇒ independent FX noise.
            r_dom, r_for
                Constant rates used when path arrays are not supplied.
            store_Z_tensor 
                If True and m > 1, save the full correlated normal tensor as
            `   paths['Z_corr']`` for downstream use.  Set False to save memory.
        Returns:
            dict: {
                "time": ndarray of time points,
                "W": Wiener paths (n_paths x (n_steps+1)),
                "X": log-spot paths (n_paths x (n_steps+1)),
                "S": spot paths (n_paths x (n_steps+1))
                "Z_corr": 
            }
        """

        # Pre-draw full normal matrix Z, antithetic pairing + moment matching normals
        half = (self.n_paths+1) // 2    # Round up when number of paths is odd
        Z_half = self.rng.standard_normal(size=(half, self.n_steps))
        Z_fx = np.vstack([ Z_half, -Z_half ])[: self.n_paths]   #trim to n_paths rows 

        # Column-wise moment matching. Making sure that samples from normal have mean 0 and variance 1
        if self.n_paths > 1: 
            col_mean = Z_fx.mean(axis=0, keepdims=True)
            col_std = Z_fx.std(axis=0, keepdims=True)
            safe_std = np.where(col_std > 0, col_std, 1.0)
            Z_fx = (Z_fx - col_mean) / safe_std


        # Pre-allocate arrays
        X = np.zeros((self.n_paths, self.n_steps+1))
        W = np.zeros_like(X)
        S = np.zeros_like(X)
        time = np.zeros(self.n_steps+1)

        # Initial log-spot
        X[:, 0] = np.log(self.S_0)
        # number of factors (first column = FX)
        if corr_L is None:
            m = 1
            store_Z = None
        else:
            m = corr_L.shape[0]
            store_Z = np.zeros((self.n_paths, self.n_steps, m))


        # Time-stepping loop (build time[i+1] at end of each iteration)
        for i in range(0, self.n_steps):
            if corr_L is not None:
                Z_uncorr = np.random.normal(loc=0.0, scale=1.0,size=(self.n_paths, m))
                Z_uncorr[:, 0] = Z_fx[:, i]
                Z_corr = Z_uncorr @ corr_L.T
                if store_Z_tensor is not None:
                    store_Z[:,i,:] = Z_corr
                # Wiener increment
                W[:, i+1] = W[:, i] + np.sqrt(self.dt) * Z_corr[:, 0]
            else:
                W[:, i+1] = W[:, i] + np.sqrt(self.dt) * Z_fx[:, i]

            # select rates for this step
            if r_dom_paths is not None and r_for_paths is not None:
                rd = r_dom_paths[:, i]
                rf = r_for_paths[:, i]
            elif r_dom is not None and r_for is not None:
                rd = r_dom
                rf = r_for
            else:
                raise ValueError("Specify scalar rates or rate paths.")

            # Euler-Maruyama on log-spot
            drift = (rd - rf - 0.5 * self.sigma**2) * self.dt
            X[:, i+1] = X[:, i] + drift + self.sigma * (W[:,i+1]-W[:,i])

            # Build time grid
            time[i+1] = time[i] + self.dt

        # Exponentiate to get spot paths
        S = np.exp(X)

        # Store results
        self.paths = {"time": time, "W": W, "X": X,"S": S}
        if store_Z_tensor is not None:
            self.paths["Z_corr"] = store_Z

        return self.paths

    def plot_paths(self, n_plot=10):

        """
        Plot simulated FX spot paths.
        """

        if self.paths is None:
            raise ValueError("Call generate_paths_with_rates() first.")

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