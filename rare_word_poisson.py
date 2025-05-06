import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Simulation parameters
N_docs = 2000
L = 15000          # tokens per document
p = 1e-4           # rare-word probability

# Simulate occurrences via Binomial (approx Poisson)
counts = np.random.binomial(L, p, size=N_docs)

# Theoretical Poisson parameter
lam = L * p

# Empirical histogram
values, freqs = np.unique(counts, return_counts=True)
freqs = freqs / N_docs

# Plot
plt.bar(values, freqs, alpha=0.6, label='Empirical')
plt.plot(values, poisson.pmf(values, lam), 'o-', label=f'Poisson(λ={lam:.1f})')
plt.xlabel('Count per document')
plt.ylabel('Probability')
plt.title('Rare-Word Occurrence: Empirical vs Poisson')
plt.legend()
plt.show()