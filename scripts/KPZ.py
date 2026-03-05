import numpy as np


def simulate_kpz(**kwargs):
    """
    1D KPZ interface simulation (periodic, Euler-Maruyama).

    Returns dict with keys: x, t, H (snapshots array, shape (n_snaps, N)).
    """
    N          = kwargs.get("N",          256)
    L          = kwargs.get("L",          128.0)
    nu         = kwargs.get("nu",         1.0)
    lam        = kwargs.get("lam",        1.0)
    D          = kwargs.get("D",          1.0)
    dt         = kwargs.get("dt",         0.01)
    nsteps     = kwargs.get("nsteps",     20000)
    save_every = kwargs.get("save_every", 200)
    seed       = kwargs.get("seed",       0)

    rng = np.random.default_rng(seed)
    dx  = L / N
    h   = np.zeros(N)
    noise_scale = np.sqrt(2.0 * D * dt / dx)

    def ddx(f):  return (np.roll(f, -1) - np.roll(f,  1)) / (2 * dx)
    def lap(f):  return (np.roll(f, -1) - 2*f + np.roll(f, 1)) / dx**2

    times, snapshots = [], []
    for step in range(nsteps + 1):
        if step % save_every == 0:
            times.append(step * dt)
            snapshots.append(h.copy())
        hx  = ddx(h)
        hxx = lap(h)
        h  += dt * (nu * hxx + 0.5 * lam * hx**2) + noise_scale * rng.standard_normal(N)
        h  -= h.mean()

    x = np.linspace(0, L, N, endpoint=False)
    return dict(x=x, t=np.array(times), H=np.array(snapshots))


def plot_kpz(result, outpath=None, colors=None):
    """3D surface of KPZ interface h(x,t), coloured by height."""
    if colors is None:
        colors = {"fuBlue": "#003366", "citeViolet": "#8000ff"}
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap


    # Colormap: dark fuBlue (low h) → light grey (high h)
    cmap = LinearSegmentedColormap.from_list(
        "kpz", [colors["fuBlue"], "#f6f4f4"]
    )

    x, t, H = result["x"], result["t"], result["H"]
    X, T    = np.meshgrid(x, t)

    # Normalise height to [0, 1] for face-colour mapping
    H_norm = (H - H.min()) / (H.max() - H.min() + 1e-15)
    face_colors = cmap(H_norm)

    fig = plt.figure(figsize=(5.8, 4.5))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X, T, H,
        facecolors=face_colors,
        linewidth=0,
        antialiased=False,
        rcount=80, ccount=80,
    )

    ax.set_xlabel("")
    ax.set_ylabel("time", labelpad=-10)
    ax.set_zlabel("")
    ax.set_title("")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

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


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from viz_style import apply_thesis_style
    colors = apply_thesis_style()

    result = simulate_kpz()
    plot_kpz(result, colors=colors)
