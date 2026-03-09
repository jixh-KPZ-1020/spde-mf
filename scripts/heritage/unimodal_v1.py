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
    # Approximately equal-area points on S² (deterministic if rng is None)
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
# Model: Unimodal vMF target
# ----------------------------

def grad_sphere_U(x, mu, alpha):
    # ∇_{S²} U(x) = -alpha * P(x) mu
    mu = mu.reshape(1, 3)
    return -alpha * tangent_project(x, mu)

def drift(x, mu, alpha):
    # overdamped Langevin: dX = -∇U dt + sqrt(2/beta) ∘ dB_{S²}
    return -grad_sphere_U(x, mu, alpha)

def target_density_vmf(x, mu, kappa_target):
    # pi(x) = c(kappa) * exp(kappa * mu·x)
    mu = mu.reshape(1, 3)
    dots = (x @ mu.T).reshape(-1)
    return vmf_norm_const(kappa_target) * np.exp(kappa_target * dots)


# ----------------------------
# Simulation (Stratonovich-style tangent noise + retraction)
# ----------------------------

def simulate_sde_sphere(
    n_particles=20000,
    T=4.0,
    dt=1e-3,
    beta=4.0,
    mu=np.array([0.0, 0.0, 1.0]),
    alpha=2.0,
    seed=0,
    save_times=None,
):
    """
    Simulates intrinsic overdamped Langevin on S² targeting pi ∝ exp(beta*alpha mu·x).

    Returns:
      saved: dict {t: X_t} with X_t shape (n_particles, 3) for requested save_times
    """
    rng = np.random.default_rng(seed)
    mu = unit(mu.reshape(1, 3)).reshape(3)

    n_steps = int(np.round(T / dt))
    X = fibonacci_sphere(n_particles, rng=rng)

    if save_times is None:
        save_times = [0.0, T]
    save_times = np.array(sorted(set(float(t) for t in save_times)))
    save_steps = {int(np.round(t / dt)): t for t in save_times}

    saved = {}
    for k in range(n_steps + 1):
        if k in save_steps:
            saved[save_steps[k]] = X.copy()

        if k == n_steps:
            break

        b = drift(X, mu, alpha)
        dW = rng.normal(size=X.shape) * np.sqrt(dt)
        dW_tan = tangent_project(X, dW)
        X_star = X + b * dt + np.sqrt(2.0 / beta) * dW_tan
        X = unit(X_star)

    return saved


# ----------------------------
# Density estimation + KL on an equal-area grid
# ----------------------------

def kde_vmf_density(eval_pts, samples, kappa_kde):
    """
    vMF kernel density on S²:
      rho_hat(x) = (1/N) Σ c(kappa_kde) exp(kappa_kde x·X_n)
    """
    c = vmf_norm_const(kappa_kde)
    dots = eval_pts @ samples.T
    K = np.exp(kappa_kde * dots)
    return c * K.mean(axis=1)

def kl_on_grid(rho, pi, area_weights):
    eps = 1e-15
    rho = np.maximum(rho, eps)
    pi  = np.maximum(pi,  eps)
    return np.sum(rho * (np.log(rho) - np.log(pi)) * area_weights)

def compute_kl_timeseries(saved_states, grid_pts, mu, beta, alpha, kappa_kde=40.0):
    """Computes KL(rho_t || pi) at the times present in saved_states."""
    mu = unit(mu.reshape(1, 3)).reshape(3)
    kappa_target = beta * alpha

    M = grid_pts.shape[0]
    area_weights = np.full(M, 4.0 * np.pi / M)

    pi = target_density_vmf(grid_pts, mu, kappa_target)

    times = np.array(sorted(saved_states.keys()))
    kls = []
    for t in times:
        X = saved_states[t]
        rho = kde_vmf_density(grid_pts, X, kappa_kde=kappa_kde)
        kls.append(kl_on_grid(rho, pi, area_weights))
    return times, np.array(kls), pi


# ----------------------------
# Public simulation wrapper (standalone / library use)
# ----------------------------

def simulate_unimodal(
    beta=4.0,
    alpha=2.0,
    mu=None,
    T=3.0,
    dt=2e-3,
    n_particles=30000,
    grid_M=6000,
    kappa_kde=35.0,
    seed=0,
):
    """
    Run simulation and package all arrays needed for plotting.

    Returns dict: grid_pts, times, kls, log_pi, log_ratio_final
    """
    if mu is None:
        mu = np.array([0.0, 0.0, 1.0])

    save_times = np.linspace(0.0, T, 31)
    saved = simulate_sde_sphere(
        n_particles=n_particles, T=T, dt=dt,
        beta=beta, mu=mu, alpha=alpha, seed=seed, save_times=save_times,
    )

    grid_pts = fibonacci_sphere(grid_M)
    times, kls, pi_grid = compute_kl_timeseries(
        saved, grid_pts, mu=mu, beta=beta, alpha=alpha, kappa_kde=kappa_kde,
    )

    X_final = saved[times[-1]]
    rho_final = kde_vmf_density(grid_pts, X_final, kappa_kde=kappa_kde)
    eps = 1e-15
    log_ratio_final = (np.log(np.maximum(rho_final, eps))
                       - np.log(np.maximum(pi_grid, eps)))
    log_pi = np.log(np.maximum(pi_grid, eps))

    return {
        "grid_pts": grid_pts,
        "times": times,
        "kls": kls,
        "log_pi": log_pi,
        "log_ratio_final": log_ratio_final,
    }


# ----------------------------
# Visualization helpers
# ----------------------------

def sphere_scatter(ax, pts, values, title="", s=5):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=values, s=s)
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()


def plot_unimodal(result, outpath, colors):
    """3-panel: log π, log(ρ̂/π) at final time, KL decay curve."""
    import matplotlib.pyplot as plt

    r = result
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131, projection="3d")
    sphere_scatter(ax1, r["grid_pts"], r["log_pi"],
                   title=r"$\log \pi(x)$ (target)", s=6)

    ax2 = fig.add_subplot(132, projection="3d")
    sphere_scatter(ax2, r["grid_pts"], r["log_ratio_final"],
                   title=r"$\log(\hat\rho/\pi)$ at final time", s=6)

    ax3 = fig.add_subplot(133)
    ax3.plot(r["times"], r["kls"])
    ax3.set_xlabel("time")
    ax3.set_ylabel(r"KL($\rho_t \| \pi$)")
    ax3.set_title("Free energy decay (relative entropy)")

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
    result = simulate_unimodal()
    plot_unimodal(result, "unimodal.pdf", colors)
    print("Saved unimodal.pdf")
    plt.show()


if __name__ == "__main__":
    main()
