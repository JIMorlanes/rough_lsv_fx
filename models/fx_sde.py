"""
Hybrid Rough-Local-Stochastic Volatility FX simulator core module.
Contains FXSimulator class for Garman-Kohlhagen spot path generation.
"""

import numpy as np
import matplotlib.pyplot as plt

class FXSimulator:
    """FX spot rate simulator using the Garman-Kohlhagen lognormal model."""

    def __init__(self, S_0, r_dom, r_for, sigma, T, n_steps, n_paths, seed=55):
        """
        Initialize FXSimulator parameters and pre-allocate arrays.

        Args:
            - S_0 (float): Initial spot FX rate.
            - r_dom (float): Domestic interest rate (annual, continuous compounding).
            - r_for (float): Foreign interest rate (annual, continuous compounding).
            - sigma (float): Volatility of the spot rate.
            - T (float): Simulation horizon in years.
            - n_steps (int): Number of discrete time steps.
            - n_paths (int): Number of Monte Carlo simulation paths.
            - seed (int, optional): Seed for NumPy RNG. Defaults to 55.
        """
        self.S_0 = S_0      
        self.r_dom = r_dom  
        self.r_for = r_for  
        self.sigma = sigma  # Constant volatility
        self.T = T         
        self.n_steps = n_steps  
        self.n_paths = n_paths
        self.seed = seed

        # Time increment
        self.dt = T / float(n_steps)
        # Define returns of simulations     
        self.paths = None   
        self.time = None    

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
        """
        Simulate FX spot paths with Euler-Maruyama discretization of GBM.

        Returns:
            dict: Dictionary with keys:
                    - 'time' (ndarray): Time grid array.
                    - 'S' (ndarray): Simulated rates of shape (n_paths, n_steps+1).
        """

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
        """
        Plot simulated FX spot paths.

        Args:
            n_plot (int, optional): Number of paths to display. Defaults to 10.

        Raises:
            ValueError: If called before `generate_paths()`.
        """

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
