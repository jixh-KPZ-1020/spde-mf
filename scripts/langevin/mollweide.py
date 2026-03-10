import numpy as np

# NOTE: do NOT import matplotlib.pyplot at module level.
# Call apply_thesis_style() before any plot function.


# ----------------------------
# Geometry helpers
# ----------------------------

def _normalize(v):
    return v / np.linalg.norm(v)

def _P(x):
    return np.eye(3) - np.outer(x, x)

def _lon_lat(x):
    lam = np.arctan2(x[1], x[0])           # longitude in [-pi, pi)
    phi = np.arcsin(np.clip(x[2], -1, 1))  # latitude  in [-pi/2, pi/2]
    return lam, phi


# ----------------------------
# Speed field m(x)
# ----------------------------

_a_eq  = 0.8
_a_lon = 0.4
_m0    = 1.0

def _m_of_x(x):
    lam, phi = _lon_lat(x)
    return _m0 * (1.0 + _a_eq * np.cos(phi)**2 + _a_lon * np.cos(2.0 * lam))

def _grad_m_ambient(x, eps=1e-6):
    g = np.zeros(3)
    for i in range(3):
        e = np.zeros(3); e[i] = 1.0
        xp = _normalize(x + eps * e)
        xm = _normalize(x - eps * e)
        g[i] = (_m_of_x(xp) - _m_of_x(xm)) / (2 * eps)
    return g


# ----------------------------
# Potential: bimodal double-well on S²
# ----------------------------

_mvec = _normalize(np.array([0.0, 0.0, 1.0]))

def _F(x):
    c = np.dot(_mvec, x)
    return 0.5 * (1.0 - c**2)

def _gradF_ambient(x):
    c = np.dot(_mvec, x)
    return -(c) * _mvec


# ----------------------------
# Simulation
# ----------------------------

def simulate_mollweide(
    beta=6.0,
    dt=5e-4,
    n_steps=600_000,
    burn_in=100_000,
    thin=20,
    seed=0,
):
    """
    Langevin on S² with variable mobility m(x).

    Returns dict: samples, hist_H, Xm, Ym, LON, LAT, Mgrid
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((n_steps + 1, 3))
    x[0] = _normalize(np.array([1.0, 0.2, 0.1]))
    sqrt_dt = np.sqrt(dt)

    for k in range(n_steps):
        xk = x[k]
        mk = _m_of_x(xk)
        gradF_tan = _P(xk) @ _gradF_ambient(xk)
        gradm_tan = _P(xk) @ _grad_m_ambient(xk)
        drift = -mk * gradF_tan + (1.0 / beta) * gradm_tan
        dB = rng.normal(size=3)
        noise = np.sqrt(2.0 * mk / beta) * (_P(xk) @ (sqrt_dt * dB))
        x[k + 1] = _normalize(xk + drift * dt + noise)

    samples = x[burn_in::thin]

    # Mollweide density histogram
    x_s, y_s, z_s = samples.T
    phi_lon = np.arctan2(y_s, x_s)
    lat = np.pi / 2 - np.arccos(np.clip(z_s, -1, 1))
    H, xedges, yedges = np.histogram2d(phi_lon, lat, bins=180, density=True)
    Xc = 0.5 * (xedges[:-1] + xedges[1:])
    Yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xm, Ym = np.meshgrid(Xc, Yc)

    # Speed-field grid
    lon  = np.linspace(-np.pi, np.pi,     361)
    latg = np.linspace(-np.pi / 2, np.pi / 2, 181)
    LON, LAT = np.meshgrid(lon, latg)
    X3 = np.cos(LAT) * np.cos(LON)
    Y3 = np.cos(LAT) * np.sin(LON)
    Z3 = np.sin(LAT)
    Mgrid = np.empty_like(X3)
    for i in range(X3.shape[0]):
        for j in range(X3.shape[1]):
            Mgrid[i, j] = _m_of_x(np.array([X3[i, j], Y3[i, j], Z3[i, j]]))

    return {
        "samples": samples,
        "hist_H": H, "Xm": Xm, "Ym": Ym,
        "LON": LON, "LAT": LAT, "Mgrid": Mgrid,
    }


# ----------------------------
# Plots
# ----------------------------

def plot_mollweide_density(result, outpath, colors):
    """Mollweide projection of the empirical invariant density."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.pcolormesh(result["Xm"], result["Ym"], result["hist_H"].T, cmap="viridis")
    ax.set_title(r"Empirical invariant density on $S^2$ (Mollweide)")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_mollweide_speed(result, outpath, colors):
    """Mollweide projection of the speed field m(lon, lat)."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.pcolormesh(result["LON"], result["LAT"], result["Mgrid"], cmap="viridis")
    ax.set_title(r"Speed field $m$ on $S^2$ (Mollweide)")
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
    result = simulate_mollweide()
    plot_mollweide_density(result, "mollweide_density.pdf", colors)
    plot_mollweide_speed(result, "mollweide_speed.pdf", colors)
    print("Saved mollweide_density.pdf and mollweide_speed.pdf")
