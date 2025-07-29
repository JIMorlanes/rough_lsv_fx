# models/fx_sde.py

import numpy as np
import matplotlib.pyplot as plt

class FXSimulator:
    def __init__(self, S_0, r_dom, r_for, sigma, T, n_steps, n_paths, seed=55):
        self.S_0 = S_0
        self.r_dom = r_dom
        self.r_for = r_for
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.dt = T / float(n_steps)
        self.seed = seed
        self.mu = r_dom - r_for
        self.paths = None
        self.time = None
        self.Z = np.random.normal(loc = 0.0, scale = 1.0, size = (self.n_paths, self.n_steps))
        self.X = np.zeros((self.n_paths, self.n_steps+1))
        self.S = np.zeros((self.n_paths, self.n_steps+1))
        self.time = np.zeros(self.n_steps+1)

    def generate_paths(self):
        np.random.seed(self.seed)
        self.X[:,0] = np.log(self.S_0)

        for i in range(0, self.n_steps):

            # Making sure that samples from a normal distribution have mean 0 and variance 1

            if self.n_paths > 1:
                self.Z[:,i] = (self.Z[:,i] - np.mean(self.Z[:,i])) / np.std(self.Z[:,i])
            self.X[:,i+1] = self.X[:,i] + (self.r_dom - self.r_for - 0.5 * self.sigma * self.sigma) * self.dt + \
                np.power(self.dt, 0.5) * self.Z[:,i]
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
