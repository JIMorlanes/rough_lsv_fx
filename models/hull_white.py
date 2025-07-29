"""
Hulll-White one-factor short-rate model under the risk-neutral measure Q:
        dr(t) = lambda*(theta - r(t)) dt + sigma dW_t
"""

import numpy as np
import matplotlib.pyplot as plt

class HullWhiteModel:
    """Class for one-factor Hull-White short-rate simulation."""

    def __init__(self, r_0, lambd, theta, sigma, T, n_steps, n_paths, seed=None):
        """
        Initialize Hull-White model parameters.

        Args:
            r0 (float): Initial short rate.
            lambd (float): Mean-reversion speed λ.
            theta (float): Long-term mean level θ.
            sigma (float): Volatility η of the short rate.
            T (float): Simulation horizon in years.
            n_steps (int): Number of discrete time steps.
            n_paths (int): Number of Monte Carlo simulation paths.
            seed (int, optional): Seed for NumPy RNG. Defaults to None.
        """
        self.r_0 = r_0
        self.lambd = lambd
        self.theta = theta
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.dt = T / float(n_steps)
        self.seed = seed
        self.paths = None
        self.time = np.linspace(0, T, n_steps + 1)

    def generate_paths(self):
        """
        Simulate short-rate paths with Euler-Maruyama discretization.

        Returns:
            dict: Dictionary with keys:
                    - 'time' (ndarray): Time grid array.
                    - 'r' (ndarray): Simulated rates of shape (n_paths, n_steps+1).
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Pre-allocate array
        r = np.zeros((self.n_paths, self.n_steps + 1))
        r[:, 0] = self.r_0

        # Simulate paths
        for i in range(self.n_steps):
            dW = np.sqrt(self.dt) * np.random.randn(self.n_paths)
            dr = self.lambd * (self.theta - r[:, i]) * self.dt + self.sigma * dW
            r[:, i + 1] = r[:, i] + dr

        self.paths = {'time': self.time, 'r': r}
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
            plt.plot(self.time, r[i], lw=0.8, alpha=0.7)
        plt.title(rf"Hull–White Short‑Rate Paths ($\lambda$={self.lambd}, $\theta$={self.theta}, $\sigma$={self.sigma})")
        plt.xlabel("Time (years)")
        plt.ylabel("Short rate $r(t)$")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
