import numpy as np

# NOTE: do NOT import matplotlib.pyplot at module level.
# Call apply_thesis_style() before any plot function.


# ----------------------------
# Model: double well + mobility
# ----------------------------

def _F(x):
    return 0.25 * x**4 - 0.5 * x**2

def _dF(x):
    return x**3 - x

def _M(x):
    # positive mobility; range ≈ [0.4, 1.6]
    return 1.0 + 0.6 * np.sin(x)

def _dM(x):
    return 0.6 * np.cos(x)

def _drift_ito(x, beta):
    # Ito form: dX = [-M dF + β⁻¹ dM] dt + sqrt(2β⁻¹M) dB
    return -_M(x) * _dF(x) + (1.0 / beta) * _dM(x)

def _sigma(x, beta):
    return np.sqrt(2.0 * (1.0 / beta) * _M(x))


# ----------------------------
# Simulation
# ----------------------------

def simulate_fdr(
    beta=6.0,
    dt=5e-4,
    n_steps=2_000_000,
    burn_in=300_000,
    thin=50,
    seed=0,
    x0=1.5,
):
    """
    Euler-Maruyama for the 1-D double-well SDE with position-dependent mobility.

    Returns dict: samples, grid, target, beta
    """
    rng = np.random.default_rng(seed)
    x = np.empty(n_steps + 1)
    x[0] = x0
    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps):
        xk = x[k]
        x[k + 1] = xk + _drift_ito(xk, beta) * dt + _sigma(xk, beta) * sqrt_dt * rng.normal()
    samples = x[burn_in::thin]

    grid = np.linspace(-2.8, 2.8, 2000)
    unnorm = np.exp(-beta * _F(grid))
    Z = np.trapz(unnorm, grid)
    return {"samples": samples, "grid": grid, "target": unnorm / Z, "beta": beta}


# ----------------------------
# Plot
# ----------------------------

def plot_fdr(result, outpath, colors):
    """Histogram of samples overlaid with target Gibbs density."""
    import matplotlib.pyplot as plt

    fuBlue = colors["fuBlue"]
    fig, ax = plt.subplots()
    ax.hist(
        result["samples"],
        bins=160,
        density=True,
        alpha=0.55,
        color=fuBlue,
        label=r"samples",
    )
    ax.plot(
        result["grid"],
        result["target"],
        color=fuBlue,
        lw=1.5,
        label=r"target $\propto e^{-\beta \mathcal{F}}$",
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


# ----------------------------
# Standalone
# ----------------------------

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from viz_style import apply_thesis_style

    colors = apply_thesis_style()
    result = simulate_fdr()
    plot_fdr(result, "double_well_fdr.pdf", colors)
    print("Saved double_well_fdr.pdf")
