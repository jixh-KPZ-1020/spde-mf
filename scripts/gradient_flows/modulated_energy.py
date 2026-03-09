import numpy as np

# Time evolution of modulated free energy (2D torus, Poisson–Drift–Diffusion)
#   -Δ φ = ρ - 1  (mean zero on torus)
#   ∂t ρ = ∇·(ρ ∇φ) + (1/β) Δρ
# Modulated free energy (relative to ρ≡1):
#   F(ρ) = ½ ∫|∇φ|² dx + (1/β) ∫(ρ log ρ - ρ + 1) dx


# ── PDE helpers (pure numpy, no pyplot) ───────────────────────────────────────

def _make_grid(n, L):
    dx = L / n
    kx = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0          # avoid ÷0; corrected in Poisson solve
    x = np.linspace(0, L, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return dict(dx=dx, cell=dx*dx, KX=KX, KY=KY, K2=K2, X=X, Y=Y)


def _solve_poisson(rhs, K2):
    Phi = np.fft.fft2(rhs) / K2
    Phi[0, 0] = 0.0
    return np.fft.ifft2(Phi).real


def _grad(a, KX, KY):
    A = np.fft.fft2(a)
    return np.fft.ifft2(1j * KX * A).real, np.fft.ifft2(1j * KY * A).real


def _free_energy(rho, beta, K2, KX, KY, cell):
    """Modulated free energy relative to ρ≡1."""
    phi  = _solve_poisson(rho - 1.0, K2)
    px, py = _grad(phi, KX, KY)
    field = 0.5 * np.sum(px**2 + py**2) * cell
    ent   = (1.0 / beta) * np.sum(rho * np.log(rho) - rho + 1.0) * cell
    return field + ent


def _step(rho, dt, beta, K2, KX, KY, n):
    phi      = _solve_poisson(rho - 1.0, K2)
    px, py   = _grad(phi, KX, KY)
    fx, fy   = rho * px, rho * py
    drift_x, _ = _grad(fx, KX, KY)
    _, drift_y  = _grad(fy, KX, KY)
    drift    = drift_x + drift_y

    R        = np.fft.fft2(rho + dt * drift)
    R       /= 1.0 + dt * (1.0 / beta) * K2
    R[0, 0]  = float(n * n)             # enforce mean = 1

    rho = np.fft.ifft2(R).real
    rho = np.clip(rho, 1e-10, None)
    rho -= rho.mean() - 1.0
    return rho


# ── Initial conditions ────────────────────────────────────────────────────────

def _torus_gauss(X, Y, cx, cy, sigma):
    dx = np.minimum(np.abs(X - cx), 1.0 - np.abs(X - cx))
    dy = np.minimum(np.abs(Y - cy), 1.0 - np.abs(Y - cy))
    return np.exp(-(dx**2 + dy**2) / (2 * sigma**2))


def _make_ics(X, Y):
    s = 0.06
    ics = {}

    # 1. Two bumps + wave (original)
    rho = (1.0
           + 0.9 * _torus_gauss(X, Y, 0.30, 0.35, s)
           + 0.7 * _torus_gauss(X, Y, 0.72, 0.68, s)
           + 0.15 * np.cos(2*np.pi*X) * np.cos(2*np.pi*Y))
    rho -= rho.mean() - 1.0
    ics["two bumps + wave"] = np.clip(rho, 1e-10, None)

    # 2. Single centred bump
    rho = 1.0 + 1.5 * _torus_gauss(X, Y, 0.5, 0.5, 0.08)
    rho -= rho.mean() - 1.0
    ics["single bump"] = np.clip(rho, 1e-10, None)

    # 3. Four symmetric bumps
    rho = (1.0
           + 0.8 * _torus_gauss(X, Y, 0.25, 0.25, s)
           + 0.8 * _torus_gauss(X, Y, 0.75, 0.25, s)
           + 0.8 * _torus_gauss(X, Y, 0.25, 0.75, s)
           + 0.8 * _torus_gauss(X, Y, 0.75, 0.75, s))
    rho -= rho.mean() - 1.0
    ics["four bumps"] = np.clip(rho, 1e-10, None)

    # 4. Pure sinusoidal (higher amplitude)
    rho = 1.0 + 0.6 * np.sin(4*np.pi*X) * np.sin(4*np.pi*Y)
    rho -= rho.mean() - 1.0
    ics["sinusoidal"] = np.clip(rho, 1e-10, None)

    return ics


# ── Simulation entry point ────────────────────────────────────────────────────

def simulate_modulated_energy(**kwargs):
    """
    Run Poisson–Drift–Diffusion on the 2D torus for several initial conditions.

    Returns dict with keys:
        curves  — list of (label, times, F_values)
        n, beta, dt, T
    """
    n    = kwargs.get("n",    192)
    beta = kwargs.get("beta", 8.0)
    dt   = kwargs.get("dt",   1.5e-4)
    T    = kwargs.get("T",    0.3)

    g     = _make_grid(n, L=1.0)
    steps = int(T / dt)
    times = np.linspace(0.0, steps * dt, steps + 1)

    ics    = _make_ics(g["X"], g["Y"])
    curves = []

    for label, rho0 in ics.items():
        print(f"  [{label}] integrating {steps} steps...")
        rho = rho0.copy()
        F   = np.empty(steps + 1)
        F[0] = _free_energy(rho, beta, g["K2"], g["KX"], g["KY"], g["cell"])
        for k in range(1, steps + 1):
            rho  = _step(rho, dt, beta, g["K2"], g["KX"], g["KY"], n)
            F[k] = _free_energy(rho, beta, g["K2"], g["KX"], g["KY"], g["cell"])
        curves.append((label, times, F))

    return dict(curves=curves, n=n, beta=beta, dt=dt, T=T)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_modulated_energy(result, outpath=None, colors=None):
    """F(t) curves for all initial conditions on one axes."""
    if colors is None:
        colors = {"fuBlue": "#003366", "citeViolet": "#8000ff"}
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    n_curves = len(result["curves"])
    cmap     = LinearSegmentedColormap.from_list(
        "thesis", [colors["citeViolet"], colors["fuBlue"]], N=n_curves
    )
    line_colors = [cmap(i / max(n_curves - 1, 1)) for i in range(n_curves)]

    fig, ax = plt.subplots()
    for (_, times, F), c in zip(result["curves"], line_colors):
        ax.plot(times, F, color=c)

    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"modulated free energy $\mathcal{F}(\rho)$")

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

    result = simulate_modulated_energy()
    plot_modulated_energy(result, colors=colors)
