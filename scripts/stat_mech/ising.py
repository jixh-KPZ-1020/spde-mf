import numpy as np

# 2D square-lattice Ising model (k_B = 1)
# Mean-field:  Tc_MF = z J,  z = 4 (square lattice)
# Exact (Onsager): Tc = 2J / ln(1 + sqrt(2)) ≈ 2.26918531421 J

TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))   # for J = 1


def _metropolis_checkerboard(L, T, n_sweeps, rng):
    """
    Vectorised checkerboard Metropolis on an L×L periodic Ising lattice.
    Updates all black (then white) sites simultaneously each half-sweep.
    """
    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    beta = 1.0 / T
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")

    for _ in range(n_sweeps):
        for parity in (0, 1):
            mask = (ii + jj) % 2 == parity
            nb = (
                np.roll(spins, 1,  axis=0) + np.roll(spins, -1, axis=0) +
                np.roll(spins, 1,  axis=1) + np.roll(spins, -1, axis=1)
            )
            dE = 2 * spins * nb
            accept = mask & (
                (dE <= 0) | (rng.random((L, L)) < np.exp(-beta * np.clip(dE, 0, None)))
            )
            spins[accept] *= -1
    return spins


def simulate_ising_tc(**kwargs):
    """
    Return J sweep, both Tc curves, and a 2D spin snapshot at Tc.
    Keyword args: J_max, n_pts, L, n_sweeps, seed.
    """
    J_max    = kwargs.get("J_max",    3.0)
    n_pts    = kwargs.get("n_pts",    400)
    L        = kwargs.get("L",        100)
    n_sweeps = kwargs.get("n_sweeps", 400)
    seed     = kwargs.get("seed",     42)

    J = np.linspace(0.0, J_max, n_pts)

    rng = np.random.default_rng(seed)
    print(f"  Simulating {L}×{L} Ising lattice at Tc ({n_sweeps} sweeps)...")
    spins = _metropolis_checkerboard(L, TC_EXACT, n_sweeps, rng)

    return dict(
        J=J,
        Tc_mf=4.0 * J,
        Tc_exact=TC_EXACT * J,
        spins=spins,
        L=L,
    )


def plot_ising_tc(result, outpath=None, colors=None):
    """Tc vs J: mean-field vs exact Onsager."""
    if colors is None:
        colors = {"fuBlue": "#003366", "citeViolet": "#8000ff"}
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(
        result["J"], result["Tc_mf"],
        color=colors["citeViolet"],
        label=r"Mean-field: $T_c^{\mathrm{MF}}=4J$",
    )
    ax.plot(
        result["J"], result["Tc_exact"],
        color=colors["fuBlue"],
        label=r"Exact (Onsager): $T_c=\frac{2J}{\ln(1+\sqrt{2})}$",
    )
    ax.set_xlabel(
        r"Coupling constant $J$ (with Boltzmann constant $k_B=1$)",
        fontsize=16,         # Changes the size
    fontweight='bold')

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath)
        print(f"  Saved → {outpath}")
    else:
        plt.show()
    plt.close(fig)


def plot_ising_lattice(result, outpath=None, colors=None):
    """2D spin snapshot at Tc."""
    if colors is None:
        colors = {"fuBlue": "#003366", "citeViolet": "#8000ff"}
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    spin_cmap = ListedColormap([colors["citeViolet"], colors["fuBlue"]])
    L = result["L"]

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(
        result["spins"], cmap=spin_cmap, vmin=-1, vmax=1,
        interpolation="nearest", origin="upper",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath)
        print(f"  Saved → {outpath}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from viz_style import apply_thesis_style
    colors = apply_thesis_style()

    result = simulate_ising_tc()
    plot_ising_tc(result, colors=colors)
    plot_ising_lattice(result, colors=colors)
