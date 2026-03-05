import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import expm_multiply

def laplacian_1d_periodic(N):
    # Discrete Laplacian with periodic boundary conditions
    main = -2.0 * np.ones(N)
    off = 1.0 * np.ones(N-1)
    L = diags([off, main, off], offsets=[-1, 0, 1], shape=(N, N), format="csr")
    # periodic wrap
    L = L.tolil()
    L[0, N-1] = 1.0
    L[N-1, 0] = 1.0
    return L.tocsr()

def gaussian_wavepacket(N, x0, sigma0, k0=0.0):
    x = np.arange(N)
    envelope = np.exp(-(x - x0)**2 / (2.0 * sigma0**2))
    phase = np.exp(1j * k0 * x)
    psi0 = envelope * phase
    psi0 = psi0 / np.linalg.norm(psi0)
    return psi0

def diagnostics(psi):
    p = np.abs(psi)**2
    p = p / p.sum()
    x = np.arange(len(p))
    mu = (x * p).sum()
    var = ((x - mu)**2 * p).sum()
    ipr = (p**2).sum()
    pr = 1.0 / ipr
    return mu, var, ipr, pr

def evolve_snapshots(H, psi0, times):
    # expm_multiply can take a sequence of times and return all snapshots efficiently
    # We want psi(t) = exp(-i H t) psi0
    A = (-1j) * H
    # expm_multiply(A, psi0, start, stop, num) returns evenly spaced times; for arbitrary times we do piecewise
    # Here we do a simple loop; for many times this is still fine for moderate N.
    psis = []
    t_prev = 0.0
    psi_prev = psi0.copy()
    for t in times:
        dt = t - t_prev
        psi_prev = expm_multiply(A * dt, psi_prev)
        psis.append(psi_prev.copy())
        t_prev = t
    return np.array(psis)

def run_comparison(N=512, T=80.0, n_steps=200, sigma0=6.0, k0=0.6, W=2.0, seed=0):
    rng = np.random.default_rng(seed)
    L = laplacian_1d_periodic(N)
    H0 = -L  # free Laplacian Hamiltonian
    V = rng.uniform(-W/2.0, W/2.0, size=N)
    HA = H0 + diags(V, 0, format="csr")

    psi0 = gaussian_wavepacket(N, x0=N//2, sigma0=sigma0, k0=k0)
    times = np.linspace(0.0, T, n_steps)

    psis0 = evolve_snapshots(H0, psi0, times)
    psisA = evolve_snapshots(HA, psi0, times)

    var0, varA = [], []
    pr0, prA = [], []
    for psi in psis0:
        _, v, _, pr = diagnostics(psi)
        var0.append(v); pr0.append(pr)
    for psi in psisA:
        _, v, _, pr = diagnostics(psi)
        varA.append(v); prA.append(pr)

    var0 = np.array(var0); varA = np.array(varA)
    pr0 = np.array(pr0); prA = np.array(prA)

    # Plot variance (spreading) and participation ratio (localization)
    fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(times, np.sqrt(var0), label="Free (H = -Δ)")
    ax[0].plot(times, np.sqrt(varA), label=f"Anderson (H = -Δ + V), W={W}")
    ax[0].set_ylabel("Std dev σ(t)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(times, pr0, label="Free (PR)")
    ax[1].plot(times, prA, label="Anderson (PR)")
    ax[1].set_xlabel("time t")
    ax[1].set_ylabel("Participation ratio PR(t)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Show final densities for visual intuition
    p_free = np.abs(psis0[-1])**2
    p_and = np.abs(psisA[-1])**2
    plt.figure(figsize=(8,3))
    plt.plot(p_free, label="Free final |ψ|^2")
    plt.plot(p_and, label="Anderson final |ψ|^2")
    plt.xlim(N//2 - 150, N//2 + 150)
    plt.ylabel("|ψ|^2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison(N=512, T=80.0, n_steps=240, sigma0=6.0, k0=0.7, W=2.5, seed=1)