import numpy as np

# NOTE: do NOT import matplotlib.pyplot at module level.
# Call apply_thesis_style() before any plot function.


# ----------------------------
# Geometry + utilities on S²
# ----------------------------

def unit(v, eps=1e-15):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

def tangent_project(x, v):
    # P(x) v = v - (x·v) x
    xv = np.sum(x * v, axis=-1, keepdims=True)
    return v - xv * x

def fibonacci_sphere(n_points, rng=None):
    # Approximately equal-area points on S²
    if rng is None:
        i = np.arange(n_points)
        u = (i + 0.5) / n_points
    else:
        u = (rng.random(n_points) + np.arange(n_points)) / n_points  # jittered stratification

    z = 1.0 - 2.0 * u
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = (np.pi * (3.0 - np.sqrt(5.0))) * np.arange(n_points)  # golden angle
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    pts = np.stack([x, y, z], axis=1)
    return unit(pts)

def vmf_norm_const(kappa):
    # vMF normalization constant on S²:
    # c(k) = k / (4π sinh k) for k>0, and 1/(4π) at k=0
    kappa = float(kappa)
    if kappa < 1e-10:
        return 1.0 / (4.0 * np.pi)
    return kappa / (4.0 * np.pi * np.sinh(kappa))


# ----------------------------
# Unimodal target: vMF via U(x) = -alpha (mu·x)
# ----------------------------

def grad_sphere_U(x, mu, alpha):
    # ∇_{S²}U = -alpha P(x)mu
    mu = mu.reshape(1, 3)
    return -alpha * tangent_project(x, mu)

def drift(x, mu, alpha):
    # overdamped Langevin drift: -∇U
    return -grad_sphere_U(x, mu, alpha)

def target_density_vmf(x, mu, kappa_target):
    mu = mu.reshape(1, 3)
    dots = (x @ mu.T).reshape(-1)
    return vmf_norm_const(kappa_target) * np.exp(kappa_target * dots)


# ----------------------------
# KDE on S² (vMF kernel) + KL
# ----------------------------

def kde_vmf_density(eval_pts, samples, kappa_kde):
    # rho_hat(x) = (1/N) Σ c(kappa) exp(kappa x·X_n)
    c = vmf_norm_const(kappa_kde)
    dots = eval_pts @ samples.T  # (M,N)
    K = np.exp(kappa_kde * dots)
    return c * K.mean(axis=1)

def kl_on_grid(rho, pi, area_weights):
    eps = 1e-15
    rho = np.maximum(rho, eps)
    pi  = np.maximum(pi,  eps)
    return np.sum(rho * (np.log(rho) - np.log(pi)) * area_weights)


# ----------------------------
# Simulation (uniform start, store snapshots, and time-average late samples)
# ----------------------------

def simulate_with_time_average(
    n_particles=20000,
    T=5.0,
    dt=2e-3,
    beta=4.0,
    mu=np.array([0.0, 0.0, 1.0]),
    alpha=2.0,
    seed=0,
    snapshot_times=(0.0, 2.5, 5.0),
    avg_window=(4.0, 5.0),
    avg_stride=10,
):
    """
    Returns:
      snapshots: dict {t: X_t} for requested snapshot_times
      avg_samples: concatenated samples from times in [avg_window[0], avg_window[1]]
                   sampled every avg_stride steps (shape (K*n_particles, 3))
    """
    rng = np.random.default_rng(seed)
    mu = unit(mu.reshape(1, 3)).reshape(3)

    n_steps = int(np.round(T / dt))
    X = fibonacci_sphere(n_particles, rng=rng)

    snap_ts = np.array(sorted(set(float(t) for t in snapshot_times)))
    snap_steps = {int(np.round(t / dt)): t for t in snap_ts}
    snapshots = {}

    w0, w1 = float(avg_window[0]), float(avg_window[1])
    w0_step = int(np.round(w0 / dt))
    w1_step = int(np.round(w1 / dt))
    avg_list = []

    for k in range(n_steps + 1):
        if k in snap_steps:
            snapshots[snap_steps[k]] = X.copy()

        if w0_step <= k <= w1_step and ((k - w0_step) % avg_stride == 0):
            avg_list.append(X.copy())

        if k == n_steps:
            break

        b = drift(X, mu, alpha)
        dW = rng.normal(size=X.shape) * np.sqrt(dt)
        dW_tan = tangent_project(X, dW)
        X_star = X + b * dt + np.sqrt(2.0 / beta) * dW_tan
        X = unit(X_star)

    avg_samples = np.concatenate(avg_list, axis=0) if len(avg_list) > 0 else X.copy()
    return snapshots, avg_samples


# ----------------------------
# Public simulation wrapper (standalone / library use)
# ----------------------------

def simulate_unimodal_v2(
    beta=4.0,
    alpha=2.0,
    mu=None,
    T=6.0,
    dt=2e-3,
    n_particles=25000,
    grid_M=7000,
    kappa_kde=35.0,
    t0=0.0,
    tm=3.0,
    tT=6.0,
    avg_window=(5.0, 6.0),
    avg_stride=10,
    seed=0,
):
    """
    Run simulation (snapshots + time-averaging + KL curve) and return plot-ready data.

    Returns dict: grid_pts, curve_times, kls, log_rho0, log_rhom, log_rho_avg,
                  log_ratio_avg, t0, tm, tT, kl0, klm, klT, avg_window
    """
    if mu is None:
        mu = np.array([0.0, 0.0, 1.0])

    snapshots, avg_samples = simulate_with_time_average(
        n_particles=n_particles, T=T, dt=dt,
        beta=beta, mu=mu, alpha=alpha, seed=0,
        snapshot_times=(t0, tm, tT),
        avg_window=avg_window, avg_stride=avg_stride,
    )

    grid_pts = fibonacci_sphere(grid_M)
    area_weights = np.full(grid_M, 4.0 * np.pi / grid_M)

    mu_u = unit(mu.reshape(1, 3)).reshape(3)
    kappa_target = beta * alpha
    pi_grid = target_density_vmf(grid_pts, mu_u, kappa_target)

    rho0   = kde_vmf_density(grid_pts, snapshots[t0], kappa_kde=kappa_kde)
    rhom   = kde_vmf_density(grid_pts, snapshots[tm], kappa_kde=kappa_kde)
    rhoT   = kde_vmf_density(grid_pts, snapshots[tT], kappa_kde=kappa_kde)
    rho_avg = kde_vmf_density(grid_pts, avg_samples,  kappa_kde=kappa_kde)

    kl0 = kl_on_grid(rho0,    pi_grid, area_weights)
    klm = kl_on_grid(rhom,    pi_grid, area_weights)
    klT = kl_on_grid(rhoT,    pi_grid, area_weights)

    # KL timeseries via a second (lighter) simulation run
    curve_times = np.linspace(0.0, T, 41)
    curve_snaps, _ = simulate_with_time_average(
        n_particles=n_particles, T=T, dt=dt,
        beta=beta, mu=mu, alpha=alpha, seed=1,
        snapshot_times=tuple(curve_times),
        avg_window=(T, T), avg_stride=1,
    )
    kls = []
    for t in curve_times:
        rho_t = kde_vmf_density(grid_pts, curve_snaps[float(t)], kappa_kde=kappa_kde)
        kls.append(kl_on_grid(rho_t, pi_grid, area_weights))
    kls = np.array(kls)

    eps = 1e-15
    log_rho0    = np.log(np.maximum(rho0,    eps))
    log_rhom    = np.log(np.maximum(rhom,    eps))
    log_rho_avg = np.log(np.maximum(rho_avg, eps))
    log_ratio_avg = log_rho_avg - np.log(np.maximum(pi_grid, eps))

    return {
        "grid_pts": grid_pts,
        "curve_times": curve_times,
        "kls": kls,
        "log_rho0": log_rho0,
        "log_rhom": log_rhom,
        "log_rho_avg": log_rho_avg,
        "log_ratio_avg": log_ratio_avg,
        "t0": t0, "tm": tm, "tT": tT,
        "kl0": kl0, "klm": klm, "klT": klT,
        "avg_window": avg_window,
    }


# ----------------------------
# Visualization helpers
# ----------------------------

def sphere_scatter(ax, pts, values, title="", s=6):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=values, s=s)
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()


def plot_unimodal_v2(result, outpath, colors):
    """4-panel: sphere densities at t0, tm, late-time avg + KL decay curve."""
    import matplotlib.pyplot as plt

    r = result
    fig = plt.figure(figsize=(14, 4))

    ax1 = fig.add_subplot(141, projection="3d")
    sphere_scatter(ax1, r["grid_pts"], r["log_rho0"],
                   title=f"Start ($t={r['t0']:g}$) $\\log\\hat{{\\rho}}$", s=6)

    ax2 = fig.add_subplot(142, projection="3d")
    sphere_scatter(ax2, r["grid_pts"], r["log_rhom"],
                   title=f"Middle ($t={r['tm']:g}$) $\\log\\hat{{\\rho}}$", s=6)

    ax3 = fig.add_subplot(143, projection="3d")
    sphere_scatter(ax3, r["grid_pts"], r["log_rho_avg"],
                   title=(f"Late-time avg $\\log\\hat{{\\rho}}$ "
                          f"($t\\in[{r['avg_window'][0]}, {r['avg_window'][1]}]$)"), s=6)

    ax4 = fig.add_subplot(144)
    ax4.plot(r["curve_times"], r["kls"], marker="o", markersize=3)
    ax4.set_xlabel("time")
    ax4.set_ylabel(r"KL($\rho_t \| \pi$)")
    ax4.set_title("Free energy decay (relative entropy)")
    ax4.scatter([r["t0"], r["tm"], r["tT"]], [r["kl0"], r["klm"], r["klT"]], zorder=3)
    ax4.annotate("start", (r["t0"], r["kl0"]), textcoords="offset points", xytext=(6, 6))
    ax4.annotate("mid",   (r["tm"], r["klm"]), textcoords="offset points", xytext=(6, 6))
    ax4.annotate("final", (r["tT"], r["klT"]), textcoords="offset points", xytext=(6, 6))

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_unimodal_v2_ratio(result, outpath, colors):
    """Convergence check: log(ρ̂/π) on sphere (flat when converged)."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    sphere_scatter(ax, result["grid_pts"], result["log_ratio_avg"],
                   title=r"Late-time avg $\log(\hat\rho/\pi)$ (flat $\Rightarrow$ converged)", s=6)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


# ----------------------------
# Standalone
# ----------------------------

def main():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from viz_style import apply_thesis_style
    import matplotlib.pyplot as plt

    colors = apply_thesis_style()
    result = simulate_unimodal_v2()
    plot_unimodal_v2(result, "unimodal_v2.pdf", colors)
    plot_unimodal_v2_ratio(result, "unimodal_v2_ratio.pdf", colors)
    print("Saved unimodal_v2.pdf and unimodal_v2_ratio.pdf")
    plt.show()


if __name__ == "__main__":
    main()
