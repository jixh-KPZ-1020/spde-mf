import numpy as np

# --- Default configuration ---
_DEFAULTS = dict(
    N=100,
    J_val=1.0,
    replicas=64,
    steps=2000,
    eq_steps=1000,
    T_high=2.0,
    T_low=0.4,
)


def simulate_SK(**kwargs):
    """
    Run SK spin-glass simulation at two temperatures.

    Returns dict with keys:
        overlaps_high, overlaps_low, T_high, T_low, N
    """
    cfg = {**_DEFAULTS, **kwargs}
    N, J_val = cfg["N"], cfg["J_val"]
    replicas, steps, eq_steps = cfg["replicas"], cfg["steps"], cfg["eq_steps"]
    T_high, T_low = cfg["T_high"], cfg["T_low"]

    J = np.random.normal(0, J_val / np.sqrt(N), (N, N))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)

    def _run(T):
        beta = 1.0 / T
        reps = np.random.choice([-1, 1], size=(replicas, N))
        print(f"  Simulating {replicas} replicas at T={T}...")
        for _ in range(steps + eq_steps):
            for r in range(replicas):
                for _ in range(N):
                    i = np.random.randint(0, N)
                    h = np.dot(J[i], reps[r])
                    dE = 2 * reps[r, i] * h
                    if dE < 0 or np.random.rand() < np.exp(-beta * dE):
                        reps[r, i] *= -1
        return reps

    def _overlaps(reps):
        q = reps @ reps.T / N
        rows, cols = np.triu_indices(replicas, k=1)
        return q[rows, cols]

    reps_high = _run(T_high)
    reps_low = _run(T_low)

    return dict(
        overlaps_high=_overlaps(reps_high),
        overlaps_low=_overlaps(reps_low),
        T_high=T_high,
        T_low=T_low,
        N=N,
    )


def plot_SK(result, outpath=None, colors=None):
    """Plot P(q) histograms for high- and low-temperature ensembles."""
    if colors is None:
        colors = {"fuBlue": "#003366", "citeViolet": "#8000ff"}
    import matplotlib.pyplot as plt

    T_high, T_low = result["T_high"], result["T_low"]
    N = result["N"]

    fig, ax = plt.subplots()

    ax.hist(
        result["overlaps_high"],
        bins=50, density=True, alpha=0.6,
        color=colors["fuBlue"],
        label=fr"High $T$ ($T={T_high}$): paramagnetic",
    )
    ax.hist(
        result["overlaps_low"],
        bins=50, density=True, alpha=0.6,
        color=colors["citeViolet"],
        label=fr"Low $T$ ($T={T_low}$): spin glass",
    )

    ax.axvline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel(r"Overlap $q_{\alpha\beta}$")
    ax.set_ylabel(r"Probability density $P(q)$")
    ax.set_title(fr"Overlap distribution $P(q)$, SK model ($N={N}$)")
    ax.legend()

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath)
        print(f"  Saved → {outpath}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from viz_style import apply_thesis_style
    colors = apply_thesis_style()

    result = simulate_SK()
    plot_SK(result, colors=colors)
