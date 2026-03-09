"""
2D periodic Anderson Hamiltonian with spatial white-noise potential.

Continuum heuristic:  H = -Δ + g ξ(x)  on T² = [0,L)² with periodic BC.
Grid regularisation:  ξ_h ~ N(0, 1/h²) cell-wise  (white-noise scaling).
Discrete model:       H_h = -Δ_h + g ξ_h  [- C_h I  if use_renorm=True]
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh


# ── PDE / linear algebra helpers ─────────────────────────────────────────────

def _periodic_laplacian(N, L):
    h = L / N
    c = 1.0 / h**2
    n = N * N
    A = lil_matrix((n, n), dtype=np.float64)

    def idx(i, j):
        return (i % N) * N + (j % N)

    for i in range(N):
        for j in range(N):
            p = idx(i, j)
            A[p, p]              =  4.0 * c
            A[p, idx(i+1, j)]   = -c
            A[p, idx(i-1, j)]   = -c
            A[p, idx(i, j+1)]   = -c
            A[p, idx(i, j-1)]   = -c
    return A.tocsr(), h


def _white_noise(N, h, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, N)) / h


def _renorm_shift(h, g, c=1.0 / (2.0 * np.pi)):
    return c * g**2 * np.log(1.0 / h)


# ── Simulation entry point ────────────────────────────────────────────────────

def simulate_ah_eigen_wn(**kwargs):
    """
    Build the 2D white-noise Anderson Hamiltonian and compute lowest eigenpairs.

    Returns dict with keys:
        evals, evecs, logdens, reals, N, L, k, h, g
    """
    N          = kwargs.get("N",          150)
    L          = kwargs.get("L",          60.0)
    g          = kwargs.get("g",          2.0)
    k          = kwargs.get("k",          4)
    seed       = kwargs.get("seed",       1)
    use_renorm = kwargs.get("use_renorm", False)
    decades    = kwargs.get("decades",    14)

    print(f"  Building {N}×{N} Anderson Hamiltonian (g={g})...")
    Lap, h = _periodic_laplacian(N, L)
    xi     = _white_noise(N, h, seed)
    C      = _renorm_shift(h, g) if use_renorm else 0.0

    H = Lap.tolil()
    H.setdiag(H.diagonal() + (g * xi).reshape(-1) - C)
    H = H.tocsr()

    print(f"  Computing {k} lowest eigenpairs...")
    evals, evecs = eigsh(H, k=k, which="SA")
    order        = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]

    # Pre-compute display arrays
    vmax    = -np.inf
    logdens, reals = [], []
    for i in range(k):
        psi = evecs[:, i].reshape(N, N)
        ld  = np.log10(psi**2 + 1e-16)
        logdens.append(ld)
        reals.append(np.real(psi))
        vmax = max(vmax, ld.max())
    vmin = vmax - decades

    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")

    return dict(
        evals=evals, evecs=evecs,
        logdens=logdens, reals=reals,
        vmin=vmin, vmax=vmax,
        X=X, Y=Y,
        N=N, L=L, k=k, h=h, g=g,
    )


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_ah_eigen_wn(result, outpath=None, colors=None):
    """Single row: log₁₀|ψ|² for each eigenfunction."""
    if colors is None:
        colors = {"fuBlue": "#003366", "citeViolet": "#8000ff"}
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cmap_log = LinearSegmentedColormap.from_list(
        "ah_log", [colors["fuBlue"], colors["citeViolet"]]
    )

    k          = result["k"]
    evals      = result["evals"]
    X, Y       = result["X"], result["Y"]
    vmin, vmax = result["vmin"], result["vmax"]
    norm_log   = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(6.0, 5.5))

    for i in range(k):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        Z  = np.clip(result["logdens"][i], vmin, vmax)
        ax.plot_surface(X, Y, Z, facecolors=cmap_log(norm_log(Z)),
                        linewidth=0, antialiased=False, shade=False,
                        rcount=60, ccount=60)
        ax.set_title(fr"$\lambda_{i+1}={evals[i]:.3f}$", fontsize=8, pad=2)
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

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

    result = simulate_ah_eigen_wn()
    plot_ah_eigen_wn(result, colors=colors)
