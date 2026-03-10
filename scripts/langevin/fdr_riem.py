import numpy as np

# NOTE: do NOT import matplotlib.pyplot at module level.
# Call apply_thesis_style() before any plot function.


# ----------------------------
# Geometry
# ----------------------------

def _normalize(v):
    return v / np.linalg.norm(v)

def _P(x):
    return np.eye(3) - np.outer(x, x)


# ----------------------------
# Model: bimodal double-well on S²
# ----------------------------

_m = _normalize(np.array([0.0, 0.0, 1.0]))

def _F(x):
    c = np.dot(_m, x)
    return 0.5 * (1.0 - c**2)

def _gradF_ambient(x):
    c = np.dot(_m, x)
    return -(c) * _m


# ----------------------------
# Simulation
# ----------------------------

def simulate_fdr_riem(
    beta=6.0,
    dt=5e-4,
    n_steps=800_000,
    burn_in=100_000,
    thin=20,
    seed=0,
):
    """
    Geometric Langevin on S² targeting exp(-beta * F) dσ (bimodal near ±m).

    Returns dict: samples, z, m, beta
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((n_steps + 1, 3))
    x[0] = _normalize(np.array([1.0, 0.2, 0.1]))
    sqrt_dt = np.sqrt(dt)
    noise_scale = np.sqrt(2.0 / beta)

    for k in range(n_steps):
        xk = x[k]
        drift = -_P(xk) @ _gradF_ambient(xk)
        dB = rng.normal(size=3)
        xkp = xk + drift * dt + noise_scale * (_P(xk) @ (sqrt_dt * dB))
        x[k + 1] = _normalize(xkp)

    samples = x[burn_in::thin]
    z = samples @ _m
    return {"samples": samples, "z": z, "m": _m.copy(), "beta": beta}


# ----------------------------
# Plot
# ----------------------------

def plot_fdr_riem(result, outpath, colors):
    """Histogram of z = m·x (should be bimodal near ±1 at equilibrium)."""
    import matplotlib.pyplot as plt

    fuBlue = colors["fuBlue"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.hist(result["z"], bins=120, density=True, alpha=0.65, color=fuBlue)
    ax.set_xlabel(r"$z = m\cdot x$")
    ax.set_ylabel("density")
    ax.set_title(r"Geometric Langevin on $S^2$: equilibrium in $z$")
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
    result = simulate_fdr_riem()
    print("E[z]   =", float(np.mean(result["z"])))
    print("E[z^2] =", float(np.mean(result["z"] ** 2)))

    import matplotlib.pyplot as plt
    plot_fdr_riem(result, "fdr_riem.pdf", colors)
    plt.show()
    print("Saved fdr_riem.pdf")
