"""
models/hull_white.py

Hulll-White one-factor short-rate model under the risk-neutral measure Q.
Provides simulation and plotting of short-rate dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt

class HullWhiteModel:
    """Class for one-factor Hull-White short-rate simulation."""

    def __init__(self, r_0, lambd, theta, eta, T, n_steps, n_paths, seed=None):
        """
        Initialize Hull-White model parameters.

        Args:
            r0 (float): Initial short rate.
            lambd (float): Mean-reversion speed λ.
            theta (float): Long-term mean level θ.
            eta (float): Volatility η of the short rate.
            T (float): Simulation horizon in years.
            n_steps (int): Number of discrete time steps.
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
        self.dt = T / float(n_steps)
        self.seed = seed
        self.paths = None

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
        Z = np.random.normal(0.0,1.0,(self.n_paths,self.n_steps))
        W = np.zeros([self.n_paths, self.n_steps+1])
        r = np.zeros((self.n_paths, self.n_steps + 1))
        r[:, 0] = self.r_0
        time = np.zeros(self.n_steps+1)

        # Simulate paths
        for i in range(0, self.n_steps):
        # making sure that samples from normal have mean 0 and variance 1
            if self.n_paths > 1:
                Z[:,i] = (Z[:,i] - Z[:,i].mean()) / Z[:,i].std()
            W[:,i+1] = W[:,i] + np.sqrt(self.dt)*Z[:,i]
            r[:,i+1] = r[:,i] + self.lambd*(self.theta - r[:,i]) * self.dt + self.eta* (W[:,i+1]-W[:,i])
            time[i+1] = time[i] + self.dt

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
        plt.title(rf"Hull-White Short-Rate Paths ($\lambda$={self.lambd}, $\theta$={self.theta}, $\eta$={self.eta})")
        plt.xlabel("Time (years)")
        plt.ylabel("Short rate $r(t)$")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_drift(self):
        """
        Plot the mean-reversion drift term over time based on the average paths.

        Raises:
            ValueError: If called before generate_paths().
        """
        if self.paths is None:
            raise ValueError("Call generate_paths() before plotting drift.")

        r = self.paths['r']
        mean_r = r.mean(axis=0)
        drift = self.lambd * (self.theta - mean_r)

        plt.figure(figsize=(10, 4))
        plt.plot(self.time, drift, lw=1.2)
        plt.title(rf"Drift Term (λ={self.lambd}, θ={self.theta})")
        plt.xlabel("Time (years)")
        plt.ylabel("Drift $λ(θ - r(t))$")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
