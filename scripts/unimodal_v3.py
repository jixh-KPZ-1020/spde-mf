import numpy as np


# ── Geometry on S² ────────────────────────────────────────────────────────────

def unit(v, eps=1e-15):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

def tangent_project(x, v):
    return v - np.sum(x * v, axis=-1, keepdims=True) * x

def fibonacci_sphere(n_points, rng=None):
    if rng is None:
        u = (np.arange(n_points) + 0.5) / n_points
    else:
        u = (rng.random(n_points) + np.arange(n_points)) / n_points
    z = 1.0 - 2.0 * u
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (3.0 - np.sqrt(5.0)) * np.arange(n_points)
    return unit(np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1))

def vmf_norm_const(kappa):
    kappa = float(kappa)
    if kappa < 1e-10:
        return 1.0 / (4.0 * np.pi)
    return kappa / (4.0 * np.pi * np.sinh(kappa))


def random_multimodal_init(
    n_particles,
    rng,
    n_modes=9,
    mode_spread=0.035,
):
    """
    Random multimodal cloud on S² from a mixture of locally concentrated blobs.
    """
    n_modes = max(2, int(n_modes))
    mode_spread = float(mode_spread)

    centers = unit(rng.normal(size=(n_modes, 3)))
    # Balanced mode populations keep several peaks visible in the first panel.
    counts = np.full(n_modes, n_particles // n_modes, dtype=int)
    remainder = n_particles - int(np.sum(counts))
    if remainder > 0:
        counts[rng.permutation(n_modes)[:remainder]] += 1

    chunks = []
    for i in range(n_modes):
        ci = counts[i]
        if ci == 0:
            continue
        blob = centers[i] + rng.normal(scale=mode_spread, size=(ci, 3))
        chunks.append(unit(blob))

    X0 = np.concatenate(chunks, axis=0)
    rng.shuffle(X0, axis=0)
    return X0


# ── vMF target ────────────────────────────────────────────────────────────────
# U(x) = -alpha (mu·x),  π(x) ∝ exp(beta·alpha · mu·x)  (vMF, κ = beta·alpha)

def drift(x, mu, alpha):
    # -∇_{S²}U = alpha · P(x) mu
    return alpha * tangent_project(x, mu.reshape(1, 3))

def target_density_vmf(grid_pts, mu, kappa_target):
    dots = (grid_pts @ mu.reshape(3, 1)).reshape(-1)
    return vmf_norm_const(kappa_target) * np.exp(kappa_target * dots)


# ── Batched vMF KDE + KL ──────────────────────────────────────────────────────

def kde_vmf_density_batched(eval_pts, samples, kappa_kde, batch=1024):
    """rho_hat(x) = (1/N) Σ c(κ) exp(κ x·Xₙ), computed in sample batches."""
    c = vmf_norm_const(kappa_kde)
    acc = np.zeros(eval_pts.shape[0], dtype=np.float64)
    for j in range(0, samples.shape[0], batch):
        acc += np.exp(kappa_kde * (eval_pts @ samples[j:j + batch].T)).sum(axis=1)
    return c * (acc / samples.shape[0])

def kl_on_grid(rho, pi, area_weights):
    eps = 1e-15
    rho = np.maximum(rho, eps)
    pi  = np.maximum(pi,  eps)
    return np.sum(rho * (np.log(rho) - np.log(pi)) * area_weights)


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_sphere_with_empirical_averaging(
    n_particles=10000,
    T=6.0,
    dt=2e-3,
    beta=4.0,
    mu=np.array([0.0, 0.0, 1.0]),
    alpha=2.0,
    seed=0,
    snapshot_times=(0.0, 6.0),
    curve_times=None,
    avg_window=(5.0, 6.0),
    avg_stride=10,
    init_modes=9,
    init_mode_spread=0.035,
    grid_pts=None,
    kappa_kde=35.0,
    kde_batch=1024,
):
    """
    Intrinsic overdamped Langevin on S² (tangent noise + retraction).
    Initial condition: random multimodal cloud on S².

    Returns (snapshots, (curve_times, kl_curve, pi_grid, rho_avg)).
    """
    rng = np.random.default_rng(seed)
    mu = unit(mu.reshape(1, 3)).reshape(3)
    n_steps = int(np.round(T / dt))

    X = random_multimodal_init(
        n_particles=n_particles,
        rng=rng,
        n_modes=init_modes,
        mode_spread=init_mode_spread,
    )

    snapshot_times = np.array(sorted(set(float(t) for t in snapshot_times)))
    snap_steps = {int(np.round(t / dt)): t for t in snapshot_times}
    snapshots = {}

    if curve_times is not None:
        curve_times = np.array(sorted(set(float(t) for t in curve_times)))
        curve_steps = {int(np.round(t / dt)): t for t in curve_times}
        kl_curve = {t: None for t in curve_times}
    else:
        curve_steps = {}
        kl_curve = None

    w0_step = int(np.round(float(avg_window[0]) / dt))
    w1_step = int(np.round(float(avg_window[1]) / dt))

    if grid_pts is None:
        raise ValueError("grid_pts must be provided.")
    M = grid_pts.shape[0]
    c_kde = vmf_norm_const(kappa_kde)
    rho_avg_acc = np.zeros(M, dtype=np.float64)
    rho_avg_count = 0

    kappa_target = beta * alpha
    pi_grid = target_density_vmf(grid_pts, mu, kappa_target)
    area_weights = np.full(M, 4.0 * np.pi / M)

    def kl_now():
        rho = kde_vmf_density_batched(grid_pts, X, kappa_kde, batch=kde_batch)
        return kl_on_grid(rho, pi_grid, area_weights)

    for k in range(n_steps + 1):
        if k in snap_steps:
            snapshots[snap_steps[k]] = X.copy()
        if k in curve_steps:
            kl_curve[curve_steps[k]] = kl_now()
        if w0_step <= k <= w1_step and (k - w0_step) % avg_stride == 0:
            for j in range(0, n_particles, kde_batch):
                rho_avg_acc += np.exp(kappa_kde * (grid_pts @ X[j:j + kde_batch].T)).sum(axis=1)
            rho_avg_count += 1
        if k == n_steps:
            break
        dW_tan = tangent_project(X, rng.normal(size=X.shape) * np.sqrt(dt))
        X = unit(X + drift(X, mu, alpha) * dt + np.sqrt(2.0 / beta) * dW_tan)

    if rho_avg_count > 0:
        rho_avg = c_kde * (rho_avg_acc / (rho_avg_count * n_particles))
    else:
        rho_avg = kde_vmf_density_batched(grid_pts, X, kappa_kde, batch=kde_batch)

    return snapshots, (curve_times, kl_curve, pi_grid, rho_avg)


# ── Public wrapper ─────────────────────────────────────────────────────────────

def simulate_unimodal_v3(
    beta=4.0,
    alpha=2.0,
    mu=None,
    T=6.0,
    dt=2e-3,
    n_particles=10000,
    grid_M=20000,
    kappa_kde=35.0,
    kde_batch=768,
    t0=0.0,
    tm=0.5,
    tT=6.0,
    avg_window=(5.0, 6.0),
    avg_stride=10,
    init_modes=9,
    init_mode_spread=0.035,
    seed=0,
):
    """
    Run simulation and return all plot-ready data.

    Result keys: grid_pts, curve_times, kls, log_rho0, log_rhom, log_rho_avg,
                 log_pi, log_ratio_avg, mu, t0, tm, tT, kl0, klm, kl_avg,
                 avg_window
    """
    if mu is None:
        mu = np.array([0.0, 0.0, 1.0])
    mu_u = unit(mu.reshape(1, 3)).reshape(3)

    curve_times = np.linspace(0.0, T, 41)
    grid_pts = fibonacci_sphere(grid_M)

    snapshots, (_, kl_curve_map, pi_grid, rho_avg) = \
        simulate_sphere_with_empirical_averaging(
            n_particles=n_particles, T=T, dt=dt,
            beta=beta, mu=mu_u, alpha=alpha, seed=seed,
            snapshot_times=(t0, tm, tT),
            curve_times=curve_times,
            avg_window=avg_window, avg_stride=avg_stride,
            init_modes=init_modes,
            init_mode_spread=init_mode_spread,
            grid_pts=grid_pts, kappa_kde=kappa_kde, kde_batch=kde_batch,
        )

    rho0 = kde_vmf_density_batched(grid_pts, snapshots[t0], kappa_kde, batch=kde_batch)
    rhom = kde_vmf_density_batched(grid_pts, snapshots[tm], kappa_kde, batch=kde_batch)

    eps = 1e-15
    log_rho0      = np.log(np.maximum(rho0,    eps))
    log_rhom      = np.log(np.maximum(rhom,    eps))
    log_rho_avg   = np.log(np.maximum(rho_avg, eps))
    log_pi        = np.log(np.maximum(pi_grid, eps))
    log_ratio_avg = log_rho_avg - np.log(np.maximum(pi_grid, eps))

    kls = np.array([kl_curve_map[float(t)] for t in curve_times])
    area_weights = np.full(grid_M, 4.0 * np.pi / grid_M)

    return {
        "grid_pts":      grid_pts,
        "curve_times":   curve_times,
        "kls":           kls,
        "log_rho0":      log_rho0,
        "log_rhom":      log_rhom,
        "log_rho_avg":   log_rho_avg,
        "log_pi":        log_pi,
        "log_ratio_avg": log_ratio_avg,
        "mu": mu_u,
        "t0": t0, "tm": tm, "tT": tT,
        "kl0":    kl_on_grid(rho0,    pi_grid, area_weights),
        "klm":    kl_on_grid(rhom,    pi_grid, area_weights),
        "kl_avg": kl_on_grid(rho_avg, pi_grid, area_weights),
        "avg_window": avg_window,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def sphere_scatter(
    ax,
    pts,
    values,
    title="",
    s=6,
    cmap="Blues",
    vmin=None,
    vmax=None,
    mu=None,
    norm=None,
    title_y=0.1,
):
    scatter_kwargs = {"c": values, "s": s, "cmap": cmap}
    if norm is not None:
        scatter_kwargs["norm"] = norm
    else:
        scatter_kwargs["vmin"] = vmin
        scatter_kwargs["vmax"] = vmax
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], **scatter_kwargs)
    _ = mu  # retained for API compatibility; no north-pole marker is drawn.
    if title:
        # Place panel caption below each sphere for a cleaner top edge.
        ax.text2D(0.5, title_y, title, transform=ax.transAxes,
                  ha="center", va="top", fontsize=17)
    ax.view_init(elev=18, azim=-55)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()


def plot_unimodal_v3(result, outpath, colors):
    """4-panel: start/intermediate/late/target densities."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fuBlue = colors["fuBlue"]
    r = result

    # Custom colormap:
    # low = fuBlue (dark), mid = gray, high = soft white.
    cmap_thesis = mcolors.LinearSegmentedColormap.from_list(
        "thesis_density_fublue_gray_softwhite", [fuBlue, "#B9C1CC", "#E9EEF5"]
    )

    def robust_limits(vals, lo=3.0, hi=97.0, soften=0.12):
        vmin = float(np.percentile(vals, lo))
        vmax = float(np.percentile(vals, hi))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
        if vmax <= vmin:
            vmax = vmin + 1.0
        # Expand limits so fewer points hit extreme colors (lower contrast).
        span = vmax - vmin
        vmin -= soften * span
        vmax += soften * span
        return vmin, vmax

    # Early-time panels: use wider limits + darker power norm to avoid
    # white saturation and make color mixture (blue/gray/white) apparent.
    vmin0, vmax0 = robust_limits(r["log_rho0"], lo=0.5, hi=99.5, soften=0.30)
    vminM, vmaxM = robust_limits(r["log_rhom"], lo=0.5, hi=99.5, soften=0.30)
    vminP, vmaxP = robust_limits(r["log_pi"])
    norm0 = mcolors.PowerNorm(gamma=1.6, vmin=vmin0, vmax=vmax0)
    normM = mcolors.PowerNorm(gamma=1.6, vmin=vminM, vmax=vmaxM)
    # Use target-based scale for late-time panel to avoid saturation from
    # empirical KDE tail outliers and keep direct visual comparability to pi.
    # Nonlinear normalization reduces "all-blue" saturation in concentrated
    # regimes while preserving ordering of density values.
    norm_target = mcolors.PowerNorm(gamma=1.8, vmin=vminP, vmax=vmaxP)
    caption_y = 0.1

    fig = plt.figure(figsize=(14, 4))

    ax1 = fig.add_subplot(141, projection="3d")
    sphere_scatter(ax1, r["grid_pts"], r["log_rho0"],
                   title=f"Arbitrary initial $\\hat{{\\mu}}({r['t0']:g})$",
                   s=6, cmap=cmap_thesis, norm=norm0, mu=r["mu"], title_y=caption_y)

    ax2 = fig.add_subplot(142, projection="3d")
    sphere_scatter(ax2, r["grid_pts"], r["log_rhom"],
                   title="",
                   s=6, cmap=cmap_thesis, norm=normM, mu=r["mu"], title_y=caption_y)

    ax3 = fig.add_subplot(143, projection="3d")
    sphere_scatter(ax3, r["grid_pts"], r["log_rho_avg"],
                   title="",
                   s=6, cmap=cmap_thesis, norm=norm_target, mu=r["mu"], title_y=caption_y)

    ax4 = fig.add_subplot(144, projection="3d")
    sphere_scatter(ax4, r["grid_pts"], r["log_pi"],
                   title=r"Gibbs $\mu^{\mathrm{inv}}$",
                   s=6, cmap=cmap_thesis, norm=norm_target, mu=r["mu"], title_y=caption_y)

    # Keep panels visually close while leaving room for below-captions.
    fig.subplots_adjust(left=0.0, right=0.99, top=0.98, bottom=0.14, wspace=-0.55)
    # Shared caption for the two middle panels.
    pos2 = ax2.get_position()
    pos3 = ax3.get_position()
    x_shared = 0.5 * (pos2.x0 + pos3.x1)
    y_shared = min(pos2.y0 + caption_y * pos2.height,
                   pos3.y0 + caption_y * pos3.height)
    fig.text(x_shared, y_shared, "Empirical Density in Evolution $\\hat{{\\mu}}(t)$",
             ha="center", va="top",fontsize=17)
    fig.savefig(outpath)
    plt.close(fig)


def plot_unimodal_v3_kl(result, outpath, colors):
    """Single-panel KL decay curve."""
    import matplotlib.pyplot as plt

    fuBlue = colors["fuBlue"]
    r = result

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r["curve_times"], r["kls"], color=fuBlue, marker="o", markersize=3)
    ax.set_xlabel("time")
    ax.set_ylabel(r"KL($\rho_t \| \pi$)")
    ax.set_title("Free energy decay")
    ax.scatter([r["t0"], r["tm"], r["tT"]], [r["kl0"], r["klm"], r["kl_avg"]],
               color=fuBlue, zorder=3)
    ax.annotate("start",
                xy=(r["t0"], r["kl0"]), xycoords="data",
                xytext=(8, -4), textcoords="offset points",
                ha="left", va="top")
    ax.annotate("mid",
                xy=(r["tm"], r["klm"]), xycoords="data",
                xytext=(6, 8), textcoords="offset points",
                ha="left", va="bottom")
    ax.annotate("late avg",
                xy=(r["tT"], r["kl_avg"]), xycoords="data",
                xytext=(-8, 8), textcoords="offset points",
                ha="right", va="bottom")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_unimodal_v3_ratio(result, outpath, colors):
    """Convergence check: log(ρ̂/π) on sphere — diverging colormap centred at 0."""
    import matplotlib.pyplot as plt
    _ = colors  # kept for consistent API signature

    vals = result["log_ratio_avg"]
    vmax = float(np.percentile(np.abs(vals), 99.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(np.abs(vals)))
    if vmax <= 0.0:
        vmax = 1.0

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    sphere_scatter(ax, result["grid_pts"], vals,
                   title=r"Late-time avg $\log(\hat\rho/\pi)$ (flat $\Rightarrow$ converged)",
                   s=6, cmap="RdBu_r", vmin=-vmax, vmax=vmax, mu=result["mu"])
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.14)
    fig.savefig(outpath)
    plt.close(fig)


# ── Standalone ────────────────────────────────────────────────────────────────

def main():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from viz_style import apply_thesis_style
    import matplotlib.pyplot as plt

    colors = apply_thesis_style()
    result = simulate_unimodal_v3()
    plot_unimodal_v3(result, "unimodal_v3.pdf", colors)
    plot_unimodal_v3_kl(result, "unimodal_v3_kl.pdf", colors)
    plot_unimodal_v3_ratio(result, "unimodal_v3_ratio.pdf", colors)
    print("Saved unimodal_v3.pdf, unimodal_v3_kl.pdf and unimodal_v3_ratio.pdf")
    plt.show()


if __name__ == "__main__":
    main()
