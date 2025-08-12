"""
models/hull_white.py

Hulll-White one-factor short-rate model under the risk-neutral measure Q.
Provides simulation and plotting of short-rate dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt

class HullWhiteModel:
    """Class for one-factor Hull-White short-rate simulation."""

    def __init__(self, r_0: float, lambd: float, theta: float, eta: float, T: float, n_steps: int, n_paths: int, 
                 seed:int|None=None) -> None:
        """
        Initialize Hull-White model parameters.

        Attributes:
            r0 (float): Initial short rate.
            lambd (float): Mean-reversion speed λ.
            theta (float): Long-term mean level θ.
            eta (float): Volatility of the short rate.
            T (float): Simulation horizon in years.
            n_steps (int): Number of discrete self.time steps.
            n_paths (int): Number of Monte Carlo simulation paths.
            seed (int, optional): Seed for NumPy RNG. Defaults to None.
        """
        self.r_0 = r_0
        self.lambd = lambd
        self.theta = theta
        self.eta = eta
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.time = np.zeros(self.n_steps+1)
        self.dt = T / float(n_steps)
        self.seed = seed
        self.paths = None

        # Local random generator
        self.rng = np.random.default_rng(seed)

        # Precompute deterministic coefficients for exact update
        self.exp_lambda_dt = np.exp(-self.lambd * self.dt)
        if self.lambd > 0:
            self.var_increment = (self.eta ** 2) / (2 * self.lambd) * (1 - np.exp(-2 * self.lambd * self.dt))
        else:
            self.var_increment = self.eta ** 2 * self.dt
        self.std_increment = np.sqrt(self.var_increment)

    def generate_paths(self) -> dict:
        """
        Simulate short-rate paths with Euler-Maruyama discretization.

        Returns:
            dict: Dictionary with keys:
                    - 'self.time' (ndarray): Time grid array.
                    - 'r' (ndarray): Simulated rates of shape (n_paths, n_steps+1).
        """

       # Antithetic variates
        half = (self.n_paths + 1) // 2
        Z_half = self.rng.standard_normal(size=(half, self.n_steps))
        Z = np.vstack((Z_half, -Z_half))[: self.n_paths]

        # Safe moment matching per time step
        if self.n_paths > 1:
            col_mean = Z.mean(axis=0, keepdims=True)
            col_std = Z.std(axis=0, keepdims=True)
            safe_std = np.where(col_std > 0, col_std, 1.0)
            Z = (Z - col_mean) / safe_std

        # Pre-allocate
        r = np.zeros((self.n_paths, self.n_steps + 1))
        r[:, 0] = self.r_0

        # Exact recurrence:
        for i in range(self.n_steps):
            deterministic = r[:, i] * self.exp_lambda_dt + self.theta * (1 - self.exp_lambda_dt)
            stochastic = self.std_increment * Z[:, i]
            r[:, i + 1] = deterministic + stochastic

        self.paths = {"time": self.time, "r": r}
        return self.paths

    def plot_paths(self, n_plot=10):
        """
        Plot simulated short-rate paths.

        Args:
            n_plot (int, optional): Number of paths to display. Defaults to 10.

        Raises:
            ValueError: If called before `generate_paths()`.
        """
        if self.paths is None:
            raise ValueError("Call generate_paths() before plotting.")

        r = self.paths['r']
        plt.figure(figsize=(10, 5))
        for i in range(min(n_plot, self.n_paths)):
            plt.plot(self.self.time, r[i], lw=0.8, alpha=0.7)
        plt.title(rf"Hull-White Short-Rate Paths ($\lambda$={self.lambd}, $\theta$={self.theta}, $\eta$={self.eta})")
        plt.xlabel("Time (years)")
        plt.ylabel("Short rate $r(t)$")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_drift(self):
        """
        Plot the mean-reversion drift term over self.time based on the average paths.

        Raises:
            ValueError: If called before generate_paths().
        """
        if self.paths is None:
            raise ValueError("Call generate_paths() before plotting drift.")

        r = self.paths['r']
        mean_r = r.mean(axis=0)
        drift = self.lambd * (self.theta - mean_r)

        plt.figure(figsize=(10, 4))
        plt.plot(self.self.time, drift, lw=1.2)
        plt.title(rf"Drift Term (λ={self.lambd}, θ={self.theta})")
        plt.xlabel("Time (years)")
        plt.ylabel("Drift $λ(θ - r(t))$")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
