import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
N = 100            # Number of spins
J_val = 1.0        # Interaction strength variance
Steps = 2000       # Monte Carlo sweeps
T_high = 2.5       # High Temperature (Paramagnetic)
T_low = 0.3        # Low Temperature (Spin Glass)

# Generate ONE fixed disorder matrix for all replicas
J_matrix = np.random.normal(0, J_val / np.sqrt(N), (N, N))
J_matrix = (J_matrix + J_matrix.T) / 2
np.fill_diagonal(J_matrix, 0)

def get_equilibrated_replicas(T):
    """Runs MC for two replicas at temperature T and returns their final states."""
    beta = 1.0 / T
    rep1 = np.random.choice([-1, 1], size=N)
    rep2 = np.random.choice([-1, 1], size=N)
    
    for _ in range(Steps):
        for rep in [rep1, rep2]:
            i = np.random.randint(0, N)
            h_i = np.dot(J_matrix[i, :], rep)
            dE = 2 * rep[i] * h_i
            
            if dE < 0 or np.random.rand() < np.exp(-beta * dE):
                rep[i] *= -1
    return rep1, rep2

# --- Run Simulations ---
print("Simulating High Temperature...")
high_rep1, high_rep2 = get_equilibrated_replicas(T_high)
high_overlap = high_rep1 * high_rep2  # Element-wise overlap

print("Simulating Low Temperature...")
low_rep1, low_rep2 = get_equilibrated_replicas(T_low)
low_overlap = low_rep1 * low_rep2     # Element-wise overlap

# --- Plotting ---
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

def plot_barcode(ax, r1, r2, overlap, title):
    # Stack the arrays: Replica 1, Replica 2, and their overlap
    data = np.vstack((r1, r2, overlap))
    
    # Plot as an image (black = -1, white = +1)
    cax = ax.imshow(data, cmap='gray', aspect='auto', interpolation='nearest')
    
    ax.set_title(title)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Replica 1', 'Replica 2', 'Overlap ($q_i$)'])
    ax.set_xlabel('Spin Index $i$')

# Plot High T
plot_barcode(axes[0], high_rep1, high_rep2, high_overlap, 
             f'High Temperature (T={T_high}): Replicas are uncorrelated')

# Plot Low T
plot_barcode(axes[1], low_rep1, low_rep2, low_overlap, 
             f'Low Temperature (T={T_low}): Replicas freeze into the same state (or inverted)')

plt.tight_layout()
plt.show()