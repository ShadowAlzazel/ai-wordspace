import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Settings
D = 1024       # embedding dimension\ n_samples = 10000
n_samples = 10000

# Generate random unit vectors
a = F.normalize(torch.randn(n_samples, D), dim=1)
b = F.normalize(torch.randn(n_samples, D), dim=1)

# Compute cosine similarities
sims = (a * b).sum(dim=1).numpy()

# Fit Gaussian parameters
mu, sigma = sims.mean(), sims.std()

# Plot histogram + PDF overlay
plt.hist(sims, bins=60, density=True, alpha=0.6)
x = np.linspace(mu-4*sigma, mu+4*sigma, 200)
plt.plot(x, norm.pdf(x, mu, sigma))
plt.title('Cosine Similarities: Empirical vs Gaussian')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.show()

# Print statistics
print(f"Empirical mean: {mu:.4f}, std: {sigma:.4f}")