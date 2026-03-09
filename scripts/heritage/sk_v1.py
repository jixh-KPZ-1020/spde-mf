import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.integrate import quad

# --- Configuration ---
N = 200            # Number of spins
J_val = 1.0        # Interaction strength variance
Steps = 3000       # Monte Carlo steps per spin
Eq_Steps = 1000    # Equilibration steps
Temps = np.linspace(2.0, 0.1, 20) # Temperature range (High to Low)

# --- Theoretical RS Solution Solver ---
def integrand(z, beta, q):
    """The integrand for the RS self-consistency equation."""
    term = np.tanh(beta * J_val * np.sqrt(q) * z)
    return np.exp(-z**2 / 2) * (term**2) / np.sqrt(2 * np.pi)

def solve_rs_q(T):
    """Numerically solve q = Integral(...) for a given T."""
    beta = 1.0 / T
    
    # Define the function f(q) = q - Integral, we want f(q) = 0
    def target_func(q):
        if q <= 0: return -0.1 # Force positive q search
        integral_val, _ = quad(integrand, -10, 10, args=(beta, q))
        return q - integral_val

    # Above Tc=1, q=0. Below Tc, q>0.
    if T >= J_val:
        return 0.0
    else:
        # Search for root in [0, 1]
        try:
            sol = root_scalar(target_func, bracket=[1e-5, 1.0], method='brentq')
            return sol.root
        except:
            return 0.0

# --- Monte Carlo Simulation (Metropolis) ---
def simulate_sk_overlap(N, T, J_matrix):
    beta = 1.0 / T
    # Create two replicas with the SAME disorder J_matrix
    replica1 = np.random.choice([-1, 1], size=N)
    replica2 = np.random.choice([-1, 1], size=N)
    
    # Precompute energy changes to speed up
    # (Naive implementation for clarity)
    
    for _ in range(Steps + Eq_Steps):
        for rep in [replica1, replica2]:
            # Pick random spin index
            i = np.random.randint(0, N)
            
            # Calculate local field h_i = sum(J_ij * s_j)
            # J_matrix is symmetric, sum over column i (or row i)
            h_i = np.dot(J_matrix[i, :], rep) - J_matrix[i, i]*rep[i]
            
            dE = 2 * rep[i] * h_i
            
            # Metropolis criterion
            if dE < 0 or np.random.rand() < np.exp(-beta * dE):
                rep[i] *= -1
        
    # Calculate Overlap q = (1/N) * sum(sigma_i^1 * sigma_i^2)
    q = np.dot(replica1, replica2) / N
    return abs(q) # We take abs because global flip symmetry allows -q or +q

# --- Main Execution ---
print(f"Simulating N={N} spins. Tc expected at T={J_val}...")

rs_solutions = []
mc_overlaps = []

# Generate ONE fixed disorder matrix for the MC simulation
# Variance of J_ij is J^2 / N
J_matrix = np.random.normal(0, J_val / np.sqrt(N), (N, N))
# Make symmetric and zero diagonal
J_matrix = (J_matrix + J_matrix.T) / 2
np.fill_diagonal(J_matrix, 0)

for T in Temps:
    # 1. Theoretical RS prediction
    q_theory = solve_rs_q(T)
    rs_solutions.append(q_theory)
    
    # 2. Monte Carlo Measurement
    # We average over a few runs to smooth noise (optional but better)
    q_sim = simulate_sk_overlap(N, T, J_matrix)
    mc_overlaps.append(q_sim)
    
    print(f"T={T:.2f} | RS Pred: {q_theory:.3f} | MC Sim: {q_sim:.3f}")

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(Temps, rs_solutions, 'r-', linewidth=2, label='RS Analytical Prediction')
plt.plot(Temps, mc_overlaps, 'bo--', label='Monte Carlo Simulation (2 Replicas)')
plt.axvline(x=J_val, color='k', linestyle=':', label='$T_c$ (Transition)')
plt.xlabel('Temperature (T)')
plt.ylabel('Edwards-Anderson Order Parameter $q$')
plt.title(f'SK Model: RS Solution vs Simulation (N={N})')
plt.gca().invert_xaxis() # Plot high T on left, low T on right
plt.legend()
plt.grid(True)
plt.show()