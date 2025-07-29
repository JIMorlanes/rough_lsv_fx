# models/fx_sde.py

import numpy as np
import matplotlib.pyplot as plt

class FXSimulator:
    
    def __init__(self, S_0, r_dom, r_for, sigma, T, n_steps, n_paths, seed=55):

        self.S_0 = S_0      # Initial spot rate (e.g. EUR/USD)
        self.r_dom = r_dom  # Domestic interest rate (e.g. USD)
        self.r_for = r_for  # Foreign interest rate (e.g. EUR)
        self.sigma = sigma  # Constant volatility
        self.T = T          # Maturity in years
        self.n_steps = n_steps  # Time steps (daily)
        self.n_paths = n_paths  # Number of simulated paths
        self.dt = T / float(n_steps)    # Time increment
        self.seed = seed    # RNG seed for reproducibility
        self.paths = None   # Hold the dict {'time':…, 'S':…} after simulation
        self.time = None    # Placeholder for the time grid array

        # Pre-draw all standard normals for efficiency 
        self.Z = np.random.normal(loc = 0.0, scale = 1.0, size = (self.n_paths, self.n_steps))

        # Pre-allocate the log-spot paths X: normally distributed increments
        # X[i, j] will be the log of the spot at step j for path i
        self.X = np.zeros((self.n_paths, self.n_steps+1))

        # Pre-allocate the spot paths S: lognormal distribution
        # S[i, j] = exp(X[i, j])
        self.S = np.zeros((self.n_paths, self.n_steps+1))

        # Build a flat time grid from 0 to T with n_steps+1 points. Useful for parameters dependent of time t
        self.time = np.zeros(self.n_steps+1)

    def generate_paths(self):

        np.random.seed(self.seed)
        self.X[:,0] = np.log(self.S_0)

        for i in range(0, self.n_steps):

            # Making sure that samples from a normal distribution have mean 0 and variance 1
            if self.n_paths > 1:
                self.Z[:,i] = (self.Z[:,i] - self.Z[:,i].mean()) / self.Z[:,i].std()
            self.X[:,i+1] = self.X[:,i] + (self.r_dom - self.r_for - 0.5 * self.sigma**2) * self.dt + \
                + self.sigma * np.sqrt(self.dt) * self.Z[:,i]
            self.time[i+1] = self.time[i] + self.dt
        
        # Compute exponent of ABM
        self.S = np.exp(self.X)
        self.paths = {"time": self.time, "S": self.S}
        return self.paths

    def plot_paths(self, n_plot = 10):

        if self.paths is None:
            raise ValueError("You must call generate_paths() first.")

        plt.figure(figsize=(10, 5))
        for i in range(min(n_plot, self.n_paths)):
            plt.plot(self.time, self.S[i], lw=0.8, alpha=0.7)
        plt.title("FX Spot Simulation – Garman-Kohlhagen Model")
        plt.xlabel("Time (years)")
        plt.ylabel("Spot Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
