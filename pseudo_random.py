import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

# Set random seed for reproducibility
#np.random.seed(42)

# Generate 1000 pseudo-random number pairs using numpy
n_points = 3500
pseudo_random_x = np.random.uniform(0, 1, n_points)
pseudo_random_y = np.random.uniform(0, 1, n_points)


# Generate 1000 quasi-random (low discrepancy) number pairs using Sobol sequence
sampler = qmc.Sobol(d=2, scramble=True, seed=42)
quasi_random_pairs = sampler.random(n_points)
quasi_random_x = quasi_random_pairs[:, 0]
quasi_random_y = quasi_random_pairs[:, 1]

# Create figure with equal-sized subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: Pseudo-random scatter plot
ax1.scatter(pseudo_random_x, pseudo_random_y, c='blue', alpha=0.6, s=10)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Pseudo-Random Distribution\n(numpy.random.uniform)')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Right plot: Quasi-random scatter plot
ax2.scatter(quasi_random_x, quasi_random_y, c='red', alpha=0.6, s=10)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Quasi-Random Distribution\n(Sobol Sequence)')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# Adjust layout and display
plt.tight_layout()
plt.show()

# Optional: Save the plot
# plt.savefig('random_comparison.png', dpi=300, bbox_inches='tight')

