import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def simulate_burgers_1d(
    N=256,              # grid points
    L=128.0,            # domain length
    nu=1.0,             # viscosity
    lam=1.0,            # nonlinearity strength (lambda)
    D=1.0,              # noise strength (consistent with KPZ: <eta eta>=2D delta delta)
    dt=0.01,            # time step
    nsteps=20000,       # number of steps
    save_every=200,     # snapshot interval
    seed=0,
    init="random"       # "random" or "sine" or "zero"
):
    rng = np.random.default_rng(seed)
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)

    # Initial condition for velocity u(x,0)
    if init == "zero":
        u = np.zeros(N)
    elif init == "random":
        u = 0.5 * rng.standard_normal(N)
        u -= u.mean()
    elif init == "sine":
        u = np.sin(2.0 * np.pi * x / L)
        u -= u.mean()
    else:
        raise ValueError("init must be 'random', 'sine', or 'zero'")

    def ddx_centered(f):
        return (np.roll(f, -1) - np.roll(f, 1)) / (2.0 * dx)

    def laplacian(f):
        return (np.roll(f, -1) - 2.0 * f + np.roll(f, 1)) / (dx * dx)

    # Noise increment for KPZ height: Δh_noise = sqrt(2D*dt/dx) * N(0,1)
    # Burgers noise consistent with mapping: Δu_noise = ∂x(Δh_noise)
    dh_noise_scale = np.sqrt(2.0 * D * dt / dx)

    times = []
    U = []

    for step in range(nsteps + 1):
        if step % save_every == 0:
            times.append(step * dt)
            U.append(u.copy())

        u_xx = laplacian(u)

        # Conservative discretization of (lam/2) * ∂x(u^2)
        u2 = u * u
        d_u2_dx = ddx_centered(u2)

        # Stochastic forcing: ∂x of KPZ white-noise increment
        dh_noise = dh_noise_scale * rng.standard_normal(N)
        du_noise = ddx_centered(dh_noise)

        # Euler–Maruyama update
        u += dt * (nu * u_xx + 0.5 * lam * d_u2_dx) + du_noise

        # Optional: remove mean (keeps u centered at 0 for plotting)
        u -= u.mean()

    return {
        "x": x,
        "times": np.array(times),
        "u_snapshots": np.array(U),
        "params": dict(N=N, L=L, nu=nu, lam=lam, D=D, dt=dt, nsteps=nsteps, save_every=save_every, seed=seed, init=init),
    }

if __name__ == "__main__":
    out = simulate_burgers_1d(
        N=256, L=128.0, nu=1.0, lam=1.0, D=1.0,
        dt=0.01, nsteps=30000, save_every=300, seed=1, init="random"
    )

    x = out["x"]
    t = out["times"]
    U = out["u_snapshots"]   # shape: (nt, N)

    # choose how many snapshots to show
    # Choose snapshots
    num_snapshots = 3
    indices = np.linspace(0, len(t) - 1, num_snapshots, dtype=int)

    # Colormap and "mix with grey" control
    cmap = cm.viridis          # your original KPZ-like colormap
    grey = np.array([0.65, 0.65, 0.65, 1.0])
    mix_to_grey = 0.35         # 0 = pure cmap, 1 = pure grey

    # Normalize colors by u value (global across all snapshots for consistency)
    norm = Normalize(vmin=np.min(U[indices]), vmax=np.max(U[indices]))

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection="3d")

    line_w = 0.6  # thinner

    for i in indices:
        xs = x
        ts = np.full_like(x, t[i])
        us = U[i]

        # Build 3D line segments: shape (N-1, 2, 3)
        pts = np.column_stack([xs, ts, us])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)

        # Color each segment by local u (use midpoint value for segment color)
        u_mid = 0.5 * (us[:-1] + us[1:])
        rgba = cmap(norm(u_mid))

        # Interpolate each segment color toward grey (same rule everywhere)
        rgba = (1.0 - mix_to_grey) * rgba + mix_to_grey * grey

        lc = Line3DCollection(segs, colors=rgba, linewidths=line_w)
        ax.add_collection3d(lc)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(t[indices].min(), t[indices].max())
    ax.set_zlim(np.min(U[indices]), np.max(U[indices]))

    ax.set_xlabel("space x")
    ax.set_ylabel("time t")
    ax.set_zlabel("Burgers field u(x,t)")
    ax.set_title("Burgers snapshots (color by u; peaks brighter; mixed with grey)")

    # Optional colorbar showing the u->color mapping (before grey mixing)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    fig.colorbar(
        mappable,
        ax=ax,          # tell matplotlib which axes to attach to
        shrink=0.55,
        aspect=14,
        pad=0.1,
        label="u (colormap mapping)"
    )

    plt.tight_layout()
    plt.show()