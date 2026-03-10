import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

# 2D cubic NLS on [0, 2π)²  —  split-step Fourier
# Tracks relative H^s norm growth for several values of s.


# ── Grid / Fourier helpers ────────────────────────────────────────────────────

def _make_grid(N, L):
    kx = fftfreq(N, d=L / (2 * np.pi * N))
    KX, KY = np.meshgrid(kx, kx)
    K2    = KX**2 + KY**2
    K_mod = np.sqrt(K2)
    dx    = L / N
    dV    = dx**2
    return dict(K2=K2, K_mod=K_mod, dV=dV, N=N)


def _get_ic(g, kind="gaussian", seed=100):
    """Single initial condition (Gaussian spectral profile by default)."""
    N, K_mod, dV = g["N"], g["K_mod"], g["dV"]
    rng = np.random.default_rng(seed)

    if kind == "gaussian":
        u_hat0 = np.exp(-K_mod**2 / 4.0) * np.exp(1j * rng.uniform(0, 2*np.pi, (N, N)))
    elif kind == "low_freq":
        u_hat0 = np.zeros((N, N), dtype=complex)
        u_hat0[K_mod <= 2.0] = 1.0
        u_hat0[0, 1] += 0.5j
        u_hat0[1, 0] -= 0.5j
    elif kind == "mid_freq":
        u_hat0 = np.zeros((N, N), dtype=complex)
        mask = (K_mod >= 3.0) & (K_mod <= 5.0)
        u_hat0[mask] = np.exp(1j * rng.uniform(0, 2*np.pi, (N, N)))[mask]

    u = ifft2(u_hat0)
    norm = np.sqrt(np.sum(np.abs(u)**2) * dV)
    return u / norm * 4.0


def _run_nls(u0, K2, dt, steps, record_every, s, dV):
    """Split-step integration; returns (times, relative H^s norm)."""
    N2   = K2.shape[0] ** 2
    Ws   = (1 + K2) ** s
    u    = u0.copy()
    half = np.exp(-1j * K2 * dt / 2.0)

    times, norms = [], []
    norm0 = None

    for i in range(steps):
        # Split-step: linear half — nonlinear — linear half
        u_hat  = fft2(u) * half
        u      = ifft2(u_hat) * np.exp(-1j * np.abs(ifft2(u_hat))**2 * dt)
        u      = ifft2(fft2(u) * half)

        if i % record_every == 0:
            u_hat_curr = fft2(u)
            Hs_sq = np.sum(Ws * np.abs(u_hat_curr)**2) / N2 * dV
            val   = float(np.sqrt(np.abs(Hs_sq)))
            if norm0 is None:
                norm0 = val if val > 0 else 1.0
            times.append(i * dt)
            norms.append(val / norm0)

    return np.array(times), np.array(norms)


# ── Simulation entry point ────────────────────────────────────────────────────

def simulate_nls(**kwargs):
    """
    Run 2D NLS split-step for several Sobolev exponents s.

    Returns dict with keys:
        curves — list of (s, times, relative_Hs_norm)
        s_values, N, dt, T_final
    """
    N            = kwargs.get("N",            64)
    L            = kwargs.get("L",            2 * np.pi)
    dt           = kwargs.get("dt",           0.005)
    T_final      = kwargs.get("T_final",      5.0)
    s_values     = kwargs.get("s_values",     [2, 4, 8])
    record_every = kwargs.get("record_every", 10)
    ic_kind      = kwargs.get("ic_kind",      "gaussian")

    g     = _make_grid(N, L)
    steps = int(T_final / dt)
    u0    = _get_ic(g, kind=ic_kind)
    curves = []

    for s in s_values:
        print(f"  NLS s={s}...")
        times, norms = _run_nls(u0, g["K2"], dt, steps, record_every, s, g["dV"])
        curves.append((s, times, norms))

    return dict(curves=curves, s_values=s_values, N=N, dt=dt, T_final=T_final)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_nls(result, outpath=None, colors=None):
    """Relative H^s norm growth for each s on a single log-scale panel."""
    if colors is None:
        colors = {"fuBlue": "#003366", "citeViolet": "#8000ff"}
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    curves   = result["curves"]
    n        = len(curves)
    cmap     = LinearSegmentedColormap.from_list(
        "thesis", [colors["citeViolet"], colors["fuBlue"]], N=n
    )
    line_colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots()
    for (_, times, norms), c in zip(curves, line_colors):
        ax.plot(times, norms, color=c)

    ax.set_yscale("log")
    ax.set_xlabel(r"time $t$", fontsize=16, fontweight='bold')
    from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=[1.0, 2.0, 5.0]))
    _sf = ScalarFormatter()
    _sf.set_scientific(False)
    _sf.set_useOffset(False)
    ax.yaxis.set_major_formatter(_sf)
    ax.yaxis.set_minor_formatter(NullFormatter())

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

    result = simulate_nls()
    plot_nls(result, colors=colors)
