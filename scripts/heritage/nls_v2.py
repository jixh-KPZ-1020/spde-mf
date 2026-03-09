import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

# --- 1. Simulation Parameters ---
N = 64                  # Grid resolution 
L = 2 * np.pi           
dt = 0.001              # Fine time step 
T_final = 5.0          
steps = int(T_final / dt)
record_every = 50       

# --- 2. Grid & Fourier Setup ---
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

kx = fftfreq(N, d=L/(2*np.pi*N)) 
ky = fftfreq(N, d=L/(2*np.pi*N))
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2

# De-aliasing mask (2/3 rule)
dealias_mask = (np.abs(KX) < N/3) & (np.abs(KY) < N/3)

# --- 3. Initial Conditions ---
# Choose your initial condition here: 'gaussian', 'dual_collision', or 'noisy_ring'
ic_choice = 'noisy_ring' 

rng = np.random.default_rng(42)

if ic_choice == 'gaussian':
    # A smooth, localized wave packet with an initial momentum phase
    u_phys = np.exp(-((X-np.pi)**2 + (Y-np.pi)**2) * 2.0) * np.exp(1j * (2*X + Y))
    u_hat0 = fft2(u_phys)

elif ic_choice == 'dual_collision':
    # Two wave packets travelling towards each other to force a nonlinear interaction
    blob1 = np.exp(-((X-np.pi+1.5)**2 + (Y-np.pi)**2) * 3.0) * np.exp(1j * 4 * X)
    blob2 = np.exp(-((X-np.pi-1.5)**2 + (Y-np.pi)**2) * 3.0) * np.exp(-1j * 4 * X)
    u_hat0 = fft2(blob1 + blob2)

elif ic_choice == 'noisy_ring':
    # A band of random phases exciting middle frequencies (cascades naturally)
    K_mod = np.sqrt(K2)
    noise = (np.exp(-((K_mod - 4)**2))) * np.exp(1j * rng.uniform(0, 2*np.pi, (N, N)))
    u_hat0 = noise

else:
    raise ValueError("Invalid ic_choice")

# Apply de-aliasing to the initial state and normalize amplitude
u_hat0 *= dealias_mask
u = ifft2(u_hat0)
u = u / np.sqrt(np.sum(np.abs(u)**2)*(L/N)**2) * 2.5 

# --- 4. Helper Functions for Exact Derivatives ---
def get_laplacian(f):
    return ifft2(-K2 * fft2(f))

def get_grad_sq(f):
    fx = ifft2(1j * KX * fft2(f))
    fy = ifft2(1j * KY * fft2(f))
    return np.abs(fx)**2 + np.abs(fy)**2

def get_ut(u):
    return 1j * get_laplacian(u) - 1j * (np.abs(u)**2 * u)

def get_utt(u, ut):
    dt_nonlin = u**2 * np.conj(ut) + 2 * np.abs(u)**2 * ut
    return 1j * get_laplacian(ut) - 1j * dt_nonlin

def get_uttt(u, ut, utt):
    term1 = 2 * u * ut * np.conj(ut) + u**2 * np.conj(utt)
    term2 = 2 * (ut * np.conj(u) + u * np.conj(ut)) * ut + 2 * np.abs(u)**2 * utt
    dt2_nonlin = term1 + term2
    return 1j * get_laplacian(utt) - 1j * dt2_nonlin

# --- 5. Metrics Storage ---
times = []
E2_vals, E4_vals, E6_vals = [], [], []

dx = L / N
dV = dx**2

print(f"Simulating P-T-V Modified Energies with IC: {ic_choice}...")

# --- 6. Time Stepping Loop ---
for i in range(steps):
    
    # Linear Step
    u_hat = fft2(u) * np.exp(-1j * K2 * dt / 2.0)
    u = ifft2(u_hat)
    
    # Nonlinear Step
    u *= np.exp(-1j * np.abs(u)**2 * dt)
    u = ifft2(fft2(u) * dealias_mask)
    
    # Linear Step
    u_hat = fft2(u) * np.exp(-1j * K2 * dt / 2.0)
    u = ifft2(u_hat)
    
    if i % record_every == 0:
        times.append(i * dt)
        
        ut = get_ut(u)
        utt = get_utt(u, ut)
        uttt = get_uttt(u, ut, utt)
        
        # --- k = 1 (E_2) ---
        norm_ut = np.sum(np.abs(ut)**2) * dV
        int_grad_u2 = 0.5 * np.sum(get_grad_sq(np.abs(u)**2)) * dV
        int_nonlin_1 = np.sum(np.abs(np.abs(u)**2 * u)**2) * dV
        E2_vals.append(norm_ut - int_grad_u2 - int_nonlin_1)
        
        # --- k = 2 (E_4) ---
        norm_utt = np.sum(np.abs(utt)**2) * dV
        dt_u2 = 2 * np.real(ut * np.conj(u))
        int_grad_dt_u2 = 0.5 * np.sum(get_grad_sq(dt_u2)) * dV
        dt_nonlin = u**2 * np.conj(ut) + 2 * np.abs(u)**2 * ut
        int_nonlin_2 = np.sum(np.abs(dt_nonlin)**2) * dV
        E4_vals.append(norm_utt - int_grad_dt_u2 - int_nonlin_2)
        
        # --- k = 3 (E_6) ---
        norm_uttt = np.sum(np.abs(uttt)**2) * dV
        dt2_u2 = 2 * np.real(utt * np.conj(u)) + 2 * np.abs(ut)**2
        int_grad_dt2_u2 = 0.5 * np.sum(get_grad_sq(dt2_u2)) * dV
        term1 = 2 * u * ut * np.conj(ut) + u**2 * np.conj(utt)
        term2 = 2 * (ut * np.conj(u) + u * np.conj(ut)) * ut + 2 * np.abs(u)**2 * utt
        dt2_nonlin = term1 + term2
        int_nonlin_3 = np.sum(np.abs(dt2_nonlin)**2) * dV
        E6_vals.append(norm_uttt - int_grad_dt2_u2 - int_nonlin_3)

print("Plotting...")

# --- 7. Visualization ---
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].plot(times, np.array(E2_vals) / E2_vals[0], 'b-', linewidth=2)
axs[0].set_title('$\mathcal{E}_2(u)$ (k=1)')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Relative Value')
axs[0].grid(True)

axs[1].plot(times, np.array(E4_vals) / E4_vals[0], 'g-', linewidth=2)
axs[1].set_title('$\mathcal{E}_4(u)$ (k=2)')
axs[1].set_xlabel('Time')
axs[1].grid(True)

axs[2].plot(times, np.array(E6_vals) / E6_vals[0], 'r-', linewidth=2)
axs[2].set_title('$\mathcal{E}_6(u)$ (k=3)')
axs[2].set_xlabel('Time')
axs[2].grid(True)

plt.suptitle(f'P-T-V Modified Energies - Initial Condition: {ic_choice}', fontsize=16)
plt.tight_layout()
plt.show()