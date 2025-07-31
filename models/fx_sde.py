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

    def __init__(self, S_0, sigma, T, n_steps, n_paths, seed=None, r_dom=None, r_for=None,
                r_dom_paths=None, r_for_paths=None):
        """
        Initialize FXSimulator.

        Args:
            S_0 (float): Initial FX spot.
            sigma (float): Spot volatility.
            T (float): Time horizon.
            n_steps (int): Steps per path.
            n_paths (int): Number of paths.
            seed (int, optional): RNG seed.
            r_dom (float, optional): Constant domestic rate.
            r_for (float, optional): Constant foreign rate.
            r_dom_paths (ndarray, optional): Rate paths (n_paths x (n_steps+1)).
            r_for_paths (ndarray, optional): Rate paths (n_paths x (n_steps+1)).
        """

        self.S_0 = S_0
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.seed = seed
        self.dt = T / float(n_steps)

        # Rate inputs: choose scalar or path arrays
        self.r_dom = r_dom
        self.r_for = r_for
        self.r_dom_paths = r_dom_paths
        self.r_for_paths = r_for_paths

        # Placeholder for outputs
        self.time = np.zeros(n_steps+1)
        self.paths = None

    def generate_paths_with_rates(self):
        
        """
        Simulate FX spot paths, storing full time grid and Wiener process.

        Returns:
            dict: {
                'time': ndarray of time points,
                'W': Wiener paths (n_paths x (n_steps+1)),
                'X': log-spot paths (n_paths x (n_steps+1)),
                'S': spot paths (n_paths x (n_steps+1))
            }
        """

        if self.seed is not None:
            np.random.seed(self.seed)

        # Pre-draw full normal matrix Z
        # antithetic pairing
        half = self.n_paths // 2
        Z_half = np.random.normal(0.0, 1.0, (half, self.n_steps))
        Z = np.vstack([ Z_half, -Z_half ])      
        # Column-wise moment matching. Making sure that samples from normal have mean 0 and variance 1
        Z -= Z.mean(axis=0, keepdims=True)
        Z /= Z.std(axis=0, keepdims=True)

        # Pre-allocate arrays
        X = np.zeros((self.n_paths, self.n_steps+1))
        W = np.zeros_like(X)
        S = np.zeros_like(X)
        time = np.zeros(self.n_steps+1)

        # Initial log-spot
        X[:, 0] = np.log(self.S_0)

        # Time-stepping loop (build time[i+1] at end of each iteration)
        for i in range(0, self.n_steps):
            # select rates for this step
            if self.r_dom_paths is not None and self.r_for_paths is not None:
                rd = self.r_dom_paths[:, i]
                rf = self.r_for_paths[:, i]
            elif self.r_dom is not None and self.r_for is not None:
                rd = self.r_dom
                rf = self.r_for
            else:
                raise ValueError("Specify scalar rates or rate paths.")

            # Wiener increment
            W[:, i+1] = W[:, i] + np.sqrt(self.dt) * Z[:, i]

            # Euler-Maruyama on log-spot
            drift = (rd - rf - 0.5 * self.sigma**2) * self.dt
            X[:, i+1] = X[:, i] + drift + self.sigma * (W[:,i+1]-W[:,i])

            # Build time grid
            time[i+1] = time[i] + self.dt

        # Exponentiate to get spot paths
        S = np.exp(X)

        # Store results
        self.paths = {"time": time, "W": W, "X": X,"S": S}

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
        plt.title(rf"FX Spot Paths – pre-draw Z ($\sigma$={self.sigma})")
        plt.xlabel("Time (years)")
        plt.ylabel("Spot Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.show()