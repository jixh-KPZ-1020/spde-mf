"""
Propagation of Chaos — Bresch-Jabin-Wang framework on T^1.

Model:
  N-particle SDE:  dX_i = (1/N) Σ_{j≠i} G'(X_i-X_j) dt + √(2σ) dW_i
  Mean-field PDE:  ∂_t ρ = -∂_x(ρ ∂_x φ) + σ ∂_xx ρ
                   -∂_xx φ = ρ - 1   (mean-zero Poisson on T^1)
  Kernel G'(x) = sign(x)/2 on (-1/2,1/2)  [W^{-1,∞}, Bresch-Jabin-Wang class]

Modulated Free Energy (KL proxy):
  H_N(t) = ∫ ρ̂_N log(ρ̂_N / ρ_pde) dx

BJW result: H_N(t) ≤ C/N at any fixed time T.

GPU: Monte Carlo batch uses PyTorch (MPS on Apple Silicon, CUDA otherwise, CPU fallback).
"""

import numpy as np


# ── 1D spectral PDE helpers ───────────────────────────────────────────────────

def _make_grid_1d(n):
    dx = 1.0 / n
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    K2 = kx ** 2
    K2[0] = 1.0          # avoid div-by-zero; DC mode corrected in Poisson solve
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    return dict(kx=kx, K2=K2, x=x, dx=dx)


def _solve_poisson_1d(rhs, K2):
    phi_hat = np.fft.fft(rhs) / K2
    phi_hat[0] = 0.0     # mean-zero potential
    return np.fft.ifft(phi_hat).real


def _grad_1d(a, kx):
    return np.fft.ifft(1j * kx * np.fft.fft(a)).real


def _step_pde_1d(rho, dt, sigma, K2, kx, n):
    phi      = _solve_poisson_1d(rho - 1.0, K2)
    dphi     = _grad_1d(phi, kx)
    flux     = rho * dphi
    dflux    = _grad_1d(flux, kx)

    R        = np.fft.fft(rho - dt * dflux)
    R       /= 1.0 + dt * sigma * K2
    R[0]     = float(n)         # enforce mean = 1
    rho      = np.fft.ifft(R).real
    rho      = np.clip(rho, 1e-10, None)
    rho     -= rho.mean() - 1.0
    return rho


def _kl_divergence(particles, rho_pde, n_bins, dx_pde):
    """KL(ρ̂_N || ρ_pde) via histogram, integrated over T."""
    counts, _ = np.histogram(particles % 1.0, bins=n_bins, range=(0.0, 1.0))
    rho_emp   = counts / (counts.sum() * (1.0 / n_bins))  # normalise to density
    # Interpolate PDE onto histogram bin centres
    bin_x    = np.linspace(0.5 / n_bins, 1.0 - 0.5 / n_bins, n_bins)
    rho_ref  = np.interp(bin_x, np.linspace(0.0, 1.0, len(rho_pde), endpoint=False), rho_pde)
    rho_ref  = np.clip(rho_ref, 1e-10, None)
    rho_emp  = np.clip(rho_emp, 1e-10, None)
    kl       = np.sum(rho_emp * np.log(rho_emp / rho_ref)) * (1.0 / n_bins)
    return max(kl, 1e-12)

def _modulated_energy(particles, rho_pde, kx, K2):
    """
    Bresch-Jabin-Wang Modulated Energy (H^{-1} distance squared).
    Measures how far the empirical measure is from the PDE density.
    """
    n_pde = len(rho_pde)
    
    # 1. Empirical Fourier coefficients: \hat{mu}_k = (1/N) * sum(exp(-i k X_j))
    # We use the kx provided by the grid (kx = 2*pi*f)
    c_k_mu = np.mean(np.exp(-1j * kx[:, None] * particles[None, :]), axis=1)

    # 2. PDE Fourier coefficients: \hat{rho}_k (normalized by n_pde for spectral consistency)
    c_k_rho = np.fft.fft(rho_pde) / n_pde

    # 3. H^{-1} norm: 0.5 * sum |diff|^2 / |k|^2
    diff = c_k_mu - c_k_rho
    
    # K2 is |k|^2. We already set K2[0]=1.0 in _make_grid_1d to avoid div-by-zero.
    # We force the DC mode difference to 0 (mass conservation).
    val = np.abs(diff)**2 / K2
    val[0] = 0.0 
    
    energy = 0.5 * np.sum(val)
    return max(energy.real, 1e-15)


# ── Numpy SDE (single trajectory, for MFE plot) ───────────────────────────────

def _run_sde_numpy(N, n_steps, dt, sigma, save_every, rng, n_pde):
    g = _make_grid_1d(n_pde)
    kx, K2 = g["kx"], g["K2"]

    # PDE Initial State: A Gaussian Bump
    rho = 1.0 + 0.8 * np.exp(-0.5 * ((g["x"] - 0.5) ** 2) / 0.04 ** 2)
    rho /= (rho.mean()) # ensure mean is 1
    
    # Sample particles from a different initial law so MFE starts away from zero.
    X = rng.uniform(0.0, 1.0, N)

    noise_scale = np.sqrt(2.0 * sigma * dt)
    times, H_vals = [0.0], [_modulated_energy(X, rho, kx, K2)]

    # Keep particle and PDE clocks synchronised: both evolve with dt each step.
    for step in range(1, n_steps + 1):
        # Force calculation...
        diff  = X[:, None] - X[None, :]
        diff -= np.round(diff)
        force = np.sign(diff).sum(axis=1) / (2.0 * N)
        X = (X + dt * force + noise_scale * rng.standard_normal(N)) % 1.0

        rho = _step_pde_1d(rho, dt, sigma, K2, kx, n_pde)

        if step % save_every == 0 or step == n_steps:
            H = _modulated_energy(X, rho, kx, K2)
            times.append(step * dt)
            H_vals.append(H)

    return np.array(times), np.array(H_vals)


# ── PyTorch batched SDE (Monte Carlo, for joint density plot) ─────────────────

def _run_sde_batch(M, N, n_steps, dt, sigma):
    """
    Run M independent SDE realisations in parallel using PyTorch.
    Returns (x1_final, x2_final) as numpy arrays of shape (M,).
    """
    import torch

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"    [batch SDE] device={device}, M={M}, N={N}, steps={n_steps}")

    X = torch.rand(M, N, device=device)

    noise_scale = (2.0 * sigma * dt) ** 0.5

    for _ in range(n_steps):
        diff   = X.unsqueeze(2) - X.unsqueeze(1)    # (M, N, N)
        diff   = diff - diff.round()                # wrap to (-0.5, 0.5)
        force  = torch.sign(diff).sum(dim=2) / (2.0 * N)   # (M, N)
        noise  = torch.randn_like(X) * noise_scale
        X      = (X + dt * force + noise) % 1.0

    X_np = X.cpu().numpy()
    return X_np[:, 0], X_np[:, 1]


# ── Simulation entry point ────────────────────────────────────────────────────

def simulate_poc(**kwargs):
    """
    Run propagation-of-chaos simulation.

    Returns dict with keys:
        mfe_curves  — list of (N, times, H_values)
        chaos_low   — (x1, x2) samples for low N
        chaos_high  — (x1, x2) samples for high N
        rho_pde_T   — PDE density at time T on grid
        grid        — 1D grid array in [0,1)
        N_low, N_high, T
    """
    N_mfe      = kwargs.get("N_mfe",      [20, 100, 500])
    N_low      = kwargs.get("N_low",      20)
    N_high     = kwargs.get("N_high",     300)
    M_monte    = kwargs.get("M_monte",    500)
    T          = kwargs.get("T",          0.5)
    dt         = kwargs.get("dt",         5e-4)
    sigma      = kwargs.get("sigma",      0.15)
    n_pde      = kwargs.get("n_pde",      256)
    save_every = kwargs.get("save_every", 50)
    seed       = kwargs.get("seed",       42)

    rng        = np.random.default_rng(seed)
    n_steps    = int(T / dt)

    # ── Plot 1: MFE decay for several N ──────────────────────────────────────
    mfe_curves = []
    for N in N_mfe:
        print(f"  [MFE] N={N}, {n_steps} steps...")
        times, H = _run_sde_numpy(N, n_steps, dt, sigma, save_every,
                                   np.random.default_rng(seed + N), n_pde)
        mfe_curves.append((N, times, H))

    # ── PDE at time T (for Plot 2 ground truth) ───────────────────────────────
    print(f"  [PDE] evolving to T={T}...")
    g   = _make_grid_1d(n_pde)
    kx, K2, dx = g["kx"], g["K2"], g["dx"]
    rho = 1.0 + 0.8 * np.exp(-0.5 * ((g["x"] - 0.5) ** 2) / 0.04 ** 2)
    rho -= rho.mean() - 1.0
    rho  = np.clip(rho, 1e-10, None)
    for _ in range(n_steps):
        rho = _step_pde_1d(rho, dt, sigma, K2, kx, n_pde)
    rho_pde_T = rho / (rho.sum() * dx)   # normalise

    # ── Plot 2: Monte Carlo joint density ─────────────────────────────────────
    monte_steps = int(T / dt)
    print(f"  [MC low N={N_low}]  M={M_monte} runs...")
    x1_low, x2_low   = _run_sde_batch(M_monte, N_low,  monte_steps, dt, sigma)
    print(f"  [MC high N={N_high}] M={M_monte} runs...")
    x1_high, x2_high = _run_sde_batch(M_monte, N_high, monte_steps, dt, sigma)

    return dict(
        mfe_curves  = mfe_curves,
        chaos_low   = (x1_low,  x2_low),
        chaos_high  = (x1_high, x2_high),
        rho_pde_T   = rho_pde_T,
        grid        = g["x"],
        N_low=N_low, N_high=N_high, T=T,
    )


# ── Plot 1: MFE Decay ─────────────────────────────────────────────────────────

def plot_poc_mfe(result, outpath=None, colors=None,
                 scale_by_N=True, normalize_at_t0=True):
    """H_N(t) curves for several N on a log-scale panel."""
    if colors is None:
        colors = {"fuBlue": "#003366", "citeViolet": "#8000ff"}
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    curves = result["mfe_curves"]
    n      = len(curves)
    cmap   = LinearSegmentedColormap.from_list(
        "poc", [colors["fuBlue"], colors["citeViolet"]], N=n
    )
    line_colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    lwidths     = [0.8, 1.4, 2.1][:n]

    fig, ax = plt.subplots()

    for (N_val, times, H), c, lw in zip(curves, line_colors, lwidths):
        H_plot = H.copy()
        if scale_by_N:
            H_plot = N_val * H_plot
        if normalize_at_t0 and H_plot[0] > 0.0:
            H_plot = H_plot / H_plot[0]
        H_plot = np.clip(H_plot, 1e-14, None)
        ax.plot(times, H_plot, color=c, linewidth=lw, label=f"$N={N_val}$")

    ax.set_yscale("log")
    ax.set_xlabel(r"time $t$")
    if scale_by_N and normalize_at_t0:
        ax.set_ylabel(r"$N\,\mathcal{H}_N(t)\,/\,N\mathcal{H}_N(0)$")
    elif scale_by_N:
        ax.set_ylabel(r"$N\,\mathcal{H}_N(t)$")
    elif normalize_at_t0:
        ax.set_ylabel(r"$\mathcal{H}_N(t)\,/\,\mathcal{H}_N(0)$")
    else:
        ax.set_ylabel(r"$\mathcal{H}_N(t)$")
    ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath)
        print(f"  Saved → {outpath}")
    else:
        plt.show()
    plt.close(fig)


# ── Plot 2: Two-Particle Joint Density ────────────────────────────────────────

def plot_poc_chaos(result, outpath=None, colors=None):
    """Three-panel heatmap: low-N joint KDE | high-N joint KDE | ρ⊗ρ."""
    if colors is None:
        colors = {"fuBlue": "#003366", "citeViolet": "#8000ff"}
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.stats import gaussian_kde

    cmap_heat = LinearSegmentedColormap.from_list(
        "poc_heat", [colors["fuBlue"], colors["citeViolet"]]
    )

    grid      = result["grid"]
    rho_T     = result["rho_pde_T"]
    N_low     = result["N_low"]
    N_high    = result["N_high"]
    x1_low,  x2_low  = result["chaos_low"]
    x1_high, x2_high = result["chaos_high"]

    # Build 2D KDE on a coarser grid for speed
    n_eval = 80
    g_eval = np.linspace(0.0, 1.0, n_eval, endpoint=False)
    GX, GY = np.meshgrid(g_eval, g_eval)
    pts    = np.vstack([GX.ravel(), GY.ravel()])

    def _kde2d(x1, x2):
        kde = gaussian_kde(np.vstack([x1, x2]))
        return kde(pts).reshape(n_eval, n_eval)

    Z_low  = _kde2d(x1_low,  x2_low)
    Z_high = _kde2d(x1_high, x2_high)
    # Tensor product: interpolate PDE ρ onto g_eval grid then outer product
    rho_eval = np.interp(g_eval, grid, rho_T)
    rho_eval = np.clip(rho_eval, 0.0, None)
    rho_eval /= rho_eval.sum() * (1.0 / n_eval)   # re-normalise
    Z_tensor = np.outer(rho_eval, rho_eval)

    panels   = [
        (Z_low,    rf"$N={N_low}$"),
        (Z_high,   rf"$N={N_high}$"),
        (Z_tensor, r"$\rho \otimes \rho$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.2))
    vmax = max(Z_low.max(), Z_high.max(), Z_tensor.max())

    for ax, (Z, title) in zip(axes, panels):
        ax.imshow(Z, origin="lower", extent=[0, 1, 0, 1],
                  cmap=cmap_heat, vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath)
        print(f"  Saved → {outpath}")
    else:
        plt.show()
    plt.close(fig)


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from viz_style import apply_thesis_style
    colors = apply_thesis_style()

    result = simulate_poc()
    plot_poc_mfe(result, colors=colors)
    plot_poc_chaos(result, colors=colors)
