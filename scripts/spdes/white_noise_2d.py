import numpy as np


def simulate_white_noise_2d(**kwargs):
    """
    White noise on a 2D periodic square [0,L)×[0,L) sampled on an Nx×Ny grid.
    If continuum_scaled=True, variance is 1/(dx·dy) to approximate continuum white noise.

    Returns dict with keys: xi, Nx, Ny, L.
    """
    Nx   = kwargs.get("Nx",   256)
    Ny   = kwargs.get("Ny",   256)
    L    = kwargs.get("L",    2.0 * np.pi)
    seed = kwargs.get("seed", 0)
    continuum_scaled = kwargs.get("continuum_scaled", True)

    rng = np.random.default_rng(seed)
    dx, dy = L / Nx, L / Ny
    sigma = 1.0 / np.sqrt(dx * dy) if continuum_scaled else 1.0
    xi = sigma * rng.standard_normal((Nx, Ny))

    return dict(xi=xi, Nx=Nx, Ny=Ny, L=L)


def plot_white_noise_2d(result, outpath=None, colors=None):
    """3D surface plot of the 2D white-noise field."""
    if colors is None:
        colors = {"fuBlue": "#003366", "citeViolet": "#8000ff"}
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    thesis_cmap = LinearSegmentedColormap.from_list(
        "thesis", [colors["citeViolet"], "white", colors["fuBlue"]]
    )

    xi = result["xi"]
    L  = result["L"]
    Nx, Ny = xi.shape

    # Downsample to ~128×128 for a legible surface
    stride = max(1, Nx // 128)
    xs = np.linspace(0, L, Nx)[::stride]
    ys = np.linspace(0, L, Ny)[::stride]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = xi[::stride, ::stride]

    fig = plt.figure(figsize=(5.8, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X, Y, Z,
        cmap=thesis_cmap,
        linewidth=0,
        antialiased=False,
        rcount=128, ccount=128,
    )

    # Clean minimal look — no labels, ticks, or title
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_title("")

    # Transparent panes for a lighter appearance
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
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from viz_style import apply_thesis_style
    colors = apply_thesis_style()

    result = simulate_white_noise_2d()
    plot_white_noise_2d(result, colors=colors)
