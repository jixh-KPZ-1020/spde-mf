This layout bridges the theoretical definitions of the Modulated Free Energy (MFE) and the asymptotic independence (chaos) with the practical reality of how to extract and plot these metrics from a discrete stochastic simulation.Visualizing Quantitative Propagation of ChaosThis document outlines the numerical and visualization pipeline for demonstrating the propagation of chaos in stochastic systems with $W^{-1,\infty}$ kernels, using the Modulated Free Energy framework.1. Plot 1: Decay of the Empirical Modulated Free EnergyConcept: The Modulated Free Energy $E_N(t)$ acts as a Lyapunov functional for the system. According to the mathematical theory (relying on Fisher information and Grönwall's inequality), $E_N(t)$ should smoothly dissipate over time, remaining bounded by a theoretical decay envelope.Visualization Strategy:A time-series line plot showing the empirically computed MFE over time. To make the quantitative estimate clear, we overlay the theoretical upper bound (the "Grönwall Tube") to demonstrate that the discrete particle system strictly respects the continuum PDE estimates.PseudocodePythonimport numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def compute_empirical_mfe_decay(particles_history, pde_history, grid, N, sigma, epsilon):
    """
    particles_history: shape (time_steps, N, d)
    pde_history: shape (time_steps, len(grid))
    """
    time_steps = len(particles_history)
    mfe_values = []
    
    for t in range(time_steps):
        X_t = particles_history[t]
        rho_pde = pde_history[t]
        
        # 1. Marginal Entropy Term (using KDE as empirical proxy)
        kde = gaussian_kde(X_t.T)
        rho_emp = kde(grid)
        
        # Avoid log(0) with a small regularization constant
        reg = 1e-10
        entropy_term = np.sum(rho_emp * np.log((rho_emp + reg) / (rho_pde + reg))) * (grid[1] - grid[0])
        
        # 2. Modulated Potential Term (Pairwise interaction)
        # Compute N x N distance matrix
        dist_matrix = compute_distance_matrix(X_t) 
        
        # Apply the specific modulated cut-off potential V_eps
        interaction_matrix = modulated_potential(dist_matrix, epsilon)
        np.fill_diagonal(interaction_matrix, 0.0) # Remove self-interaction
        
        potential_term = np.sum(interaction_matrix) / (N * (N - 1))
        
        # 3. Total MFE
        mfe_t = entropy_term + (1.0 / sigma) * potential_term
        mfe_values.append(mfe_t)
        
    return np.array(mfe_values)

def plot_mfe_decay(time_array, mfe_values, N, alpha, C1, C2):
    plt.figure(figsize=(10, 6))
    
    # Plot empirical MFE
    plt.plot(time_array, mfe_values, label=f'Empirical MFE (N={N})', color='purple', linewidth=2)
    
    # Plot Theoretical Grönwall Bound: E_N(t) <= C1 * N^{-alpha} * exp(-C2 * t)
    theoretical_bound = C1 * (N ** -alpha) * np.exp(-C2 * time_array)
    plt.plot(time_array, theoretical_bound, 'k--', label='Theoretical Grönwall Bound')
    
    # Fill the "tube"
    plt.fill_between(time_array, 0, theoretical_bound, color='gray', alpha=0.2)
    
    plt.title("Decay of Empirical Modulated Free Energy")
    plt.xlabel("Time (t)")
    plt.ylabel(r"$\mathcal{E}_N(t)$")
    plt.yscale('log') # Log-scale often makes exponential decay clearer
    plt.legend()
    plt.grid(True)
    plt.show()

2. Plot 2: Two-Particle Joint Density (Visualizing Chaos)Concept:"Propagation of chaos" literally means that as $N \to \infty$, the joint probability distribution of any $k$ particles approaches the tensor product of the macroscopic limit density. For $k=2$, this means:$$f^{(2)}_N(t, x_1, x_2) \approx \rho(t, x_1) \otimes \rho(t, x_2)$$Visualization Strategy:We will generate two side-by-side 2D heatmaps at a fixed time $T > 0$.Plot A (Low N): Displays the empirical joint density of two particles. Because $N$ is small, the singular interaction forces them into highly correlated states (the heatmap will look "smudged" or strictly clustered along the diagonal, showing dependence).Plot B (High N): Displays the same for a very large $N$. The heatmap should look like a perfect, symmetrical grid—visually identical to the outer product $\rho(x) \rho(y)$, proving that the particles have become statistically independent.Note: To plot a smooth empirical joint density $f^{(2)}_N$, you cannot just use one simulation run. You must run $M$ independent Monte Carlo simulations of the $N$-particle system and extract the positions of Particle 1 and Particle 2 from each run to build a statistical ensemble.PseudocodePythonimport numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def build_two_particle_ensemble(N, num_simulations, T, dt, sigma, epsilon):
    """
    Runs `num_simulations` independent SDE realizations up to time T.
    Extracts the final positions of the first two particles to build the joint distribution.
    """
    x1_samples = np.zeros(num_simulations)
    x2_samples = np.zeros(num_simulations)
    
    for m in range(num_simulations):
        # Run SDE solver for N particles up to time T
        final_state = run_sde_solver(N, T, dt, sigma, epsilon) 
        
        # Track specifically particle index 0 and index 1
        x1_samples[m] = final_state[0]
        x2_samples[m] = final_state[1]
        
    return x1_samples, x2_samples

def plot_propagation_of_chaos(x1_low_N, x2_low_N, x1_high_N, x2_high_N, rho_pde_T, grid):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Low N Empirical Joint Density (Shows Correlation/Dependence)
    # Perform 2D KDE on the samples
    positions_low = np.vstack([x1_low_N, x2_low_N])
    kde_low = gaussian_kde(positions_low)
    
    # Evaluate on a 2D meshgrid
    X, Y = np.meshgrid(grid, grid)
    Z_low = kde_low(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
    ax0 = axes[0].contourf(X, Y, Z_low, cmap='magma', levels=20)
    axes[0].set_title("Empirical Joint Density (Low N)\nShows particle correlation")
    axes[0].set_xlabel("$X_1$")
    axes[0].set_ylabel("$X_2$")
    fig.colorbar(ax0, ax=axes[0])

    # 2. High N Empirical Joint Density (Shows Asymptotic Independence)
    positions_high = np.vstack([x1_high_N, x2_high_N])
    kde_high = gaussian_kde(positions_high)
    Z_high = kde_high(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
    ax1 = axes[1].contourf(X, Y, Z_high, cmap='magma', levels=20)
    axes[1].set_title("Empirical Joint Density (High N)\nChaos propagates (Independence)")
    axes[1].set_xlabel("$X_1$")
    axes[1].set_ylabel("$X_2$")
    fig.colorbar(ax1, ax=axes[1])

    # 3. Ground Truth: Tensorized PDE Limit (Outer Product)
    # Z_tensor(x,y) = rho(x) * rho(y)
    Z_tensor = np.outer(rho_pde_T, rho_pde_T) 
    
    ax2 = axes[2].contourf(X, Y, Z_tensor, cmap='magma', levels=20)
    axes[2].set_title(r"Macroscopic Limit $\rho \otimes \rho$")
    axes[2].set_xlabel("$x$")
    axes[2].set_ylabel("$y$")
    fig.colorbar(ax2, ax=axes[2])

    plt.tight_layout()
    plt.show()