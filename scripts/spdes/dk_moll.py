import numpy as np
import matplotlib.ticker as ticker


# ── helpers ────────────────────────────────────────────────────────────────────

def _f_delta(x, delta):
    """C¹ approximation of sign(x)√|x| from Eq. (2.7)."""
    x  = np.asarray(x, dtype=float)
    ax = np.abs(x)
    s  = np.sign(x)
    y  = np.zeros_like(x)

    r1 = ax <= delta / 2
    r2 = (ax > delta / 2) & (ax <= delta)
    r3 = ax > delta

    y[r1] = x[r1] / np.sqrt(delta)
    y[r2] = (
        -2 * x[r2] ** 3 / delta ** 2.5
        + s[r2] * 4 * x[r2] ** 2 / delta ** 1.5
        - 3 * x[r2] / (2 * np.sqrt(delta))
        + s[r2] * np.sqrt(delta) / 2
    )
    y[r3] = s[r3] * np.sqrt(ax[r3])
    return y


def _eta_hat(xi, r=0.5, R=3.0):
    """
    Smooth cutoff: 1 for |ξ| ≤ r, 0 for |ξ| ≥ R, C∞ transition in between.
    Uses the standard smooth step built from f(t) = exp(−1/t).
    """
    xi = np.asarray(xi, dtype=float)
    t  = np.clip((np.abs(xi) - r) / (R - r), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        ft  = np.where(t > 0, np.exp(-1.0 / t),       0.0)
        f1t = np.where(t < 1, np.exp(-1.0 / (1 - t)), 0.0)
    denom = ft + f1t
    h = np.where(denom > 0, ft / denom, 0.0)   # smooth step 0 → 1
    return 1.0 - h


# ── simulate ───────────────────────────────────────────────────────────────────

def simulate_dk_moll(**kwargs):
    """
    Pure-numpy data for the two Dean-Kawasaki mollifier figures.

    Returns dict with keys:
        x        – evaluation grid for f_delta  (shape N,)
        deltas   – list of δ values
        f_curves – dict  delta -> f_delta(x, delta) array
        f_limit  – sign(x)·√|x| on x
        k        – integer Fourier modes (shape M,)
        L_values – list of L values
        theta    – dict  L -> θ_k^N(k) array
    """
    x_lim    = kwargs.get("x_lim",    0.2)
    Nx       = kwargs.get("Nx",       2000)
    deltas   = kwargs.get("deltas",   [0.2, 0.1, 0.03])
    L_2d     = kwargs.get("L_2d",     8)
    k_max_2d = kwargs.get("k_max_2d", int(4.0 * L_2d) + 2)

    x = np.linspace(-x_lim * 0.1, x_lim, Nx)
    f_curves = {d: _f_delta(x, d) for d in deltas}
    f_limit  = np.sign(x) * np.sqrt(np.abs(x))

    g        = np.arange(-k_max_2d, k_max_2d + 1)
    K1, K2   = np.meshgrid(g, g)
    knorm    = np.sqrt(K1**2 + K2**2)
    Th       = np.sqrt(_eta_hat(knorm / L_2d))

    return dict(
        x=x, deltas=deltas, f_curves=f_curves, f_limit=f_limit,
        L_2d=L_2d, k1_2d=K1.ravel(), k2_2d=K2.ravel(), theta_2d=Th.ravel(),
    )


# ── plot helpers ───────────────────────────────────────────────────────────────

def _dk_palette(colors, n):
    """n shades interpolating from citeViolet (large δ) to fuBlue (small δ)."""
    import matplotlib.colors as mc
    start = mc.to_rgb(colors["citeViolet"])   # large δ → violet
    end   = mc.to_rgb(colors["fuBlue"])       # small δ → blue
    return [
        tuple(start[c] + (end[c] - start[c]) * i / max(n - 1, 1) for c in range(3))
        for i in range(n)
    ]


# ── figure 1 : f_delta approximation ──────────────────────────────────────────

def plot_dk_fdelta(result, outpath=None, colors=None):
    """Figure 1 — C¹ regularisation f_δ(x) → sign(x)√|x|  (Eq. 2.7)."""
    if colors is None:
        colors = {"fuBlue": "#003366", "citeViolet": "#d9b7fa"}
    import matplotlib.pyplot as plt

    x        = result["x"]
    deltas   = result["deltas"]
    f_curves = result["f_curves"]
    f_limit  = result["f_limit"]

    palette = _dk_palette(colors, len(deltas))

    fig, ax = plt.subplots()

    # curves ordered large→small δ so darkest (smallest) is on top
    for d, col in zip(deltas, palette):
        ax.plot(x, f_curves[d], color=col, lw=1.4,
                label=rf"$\delta={d}$")

    ax.plot(x, f_limit, color="#888888", lw=1.8,
            linestyle="--", label=r"$\mathrm{sign}(x)\sqrt{|x|}$")

    ax.set_xlim(x[0], x[-1])
    
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5)) 
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3)) 
    
    ax.plot(0, 0, marker="o", markersize=4, color="#888888", zorder=5)

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath)
        print(f"  Saved → {outpath}")
    else:
        plt.show()
    plt.close(fig)


# ── figure 2 : spectral mollifier θ_k^N ───────────────────────────────────────

def plot_dk_theta(result, outpath=None, colors=None):
    """Figure 2 — spectral mollifier θ_k^N on Z², 3D scatter (Eq. 1.16)."""
    if colors is None:
        colors = {"fuBlue": "#003366", "citeViolet": "#d9b7fa"}

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.ticker import MultipleLocator

    L  = int(result["L_2d"])
    k1 = result["k1_2d"]
    k2 = result["k2_2d"]
    th = result["theta_2d"]

    base_rgb  = mcolors.to_rgb(colors["fuBlue"])
    light_rgb = mcolors.to_rgb(colors.get("citeViolet", "#d9b7fa"))
    rgba_colors = np.zeros((256, 4))
    t = np.linspace(0.0, 1.0, 256)
    for c in range(3):
        rgba_colors[:, c] = light_rgb[c] + t * (base_rgb[c] - light_rgb[c])
    rgba_colors[:, 3] = 1.0
    cmap = mcolors.ListedColormap(rgba_colors)

    fig = plt.figure(figsize=(5.8, 4.8))
    ax  = fig.add_subplot(111, projection="3d")

    ax.scatter(k1, k2, th,
               c=th, cmap=cmap, s=3, alpha=0.9,
               vmin=0, vmax=1, linewidths=0)

    # Lattice bounds
    k_max = int(max(abs(k1).max(), abs(k2).max()))

    # Synchronize axis bounds and grid ticks
    num_intervals = 5
    grid_step = max(1, (k_max + num_intervals - 1) // num_intervals) 
    axis_bound = grid_step * num_intervals

    ax.set_xlim(-axis_bound, axis_bound)
    ax.set_ylim(-axis_bound, axis_bound)
    ax.set_zlim(0, 1)

    ax.xaxis.set_major_locator(MultipleLocator(grid_step))
    ax.yaxis.set_major_locator(MultipleLocator(grid_step))
    ax.zaxis.set_major_locator(MultipleLocator(0.2))

    # dashed guide line at peak
    ax.plot([0, axis_bound], [0, axis_bound], [1, 1],
            color="#888888", lw=1.0, linestyle="--", zorder=0)

    # --- NEW: Two-sided arrow inside the plot ---
    # Draw the main line measuring the support (-L to L) along y=0, z=0
    ax.plot([-2.8*L, 2.8*L], [0, 0], [0, 0], color="k", lw=1.2, zorder=5)

    # Label the arrow (elevated slightly on the z-axis so it doesn't clip through the line)
    # Note: If your full support width is 2L, you might want to change the text to r"$2L$"
    ax.text(0, 0, 0.05, r"$L_N$",
            ha="center", va="bottom", fontsize=10, color="k", zorder=5)
    # --------------------------------------------

    # cosmetics
    ax.set_ylabel(r"$k=(k_1,k_2)$", labelpad=-8)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # 1. Define every place you want a tick MARK to appear
    tick_locations = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # 2. Define the exact text for each of those locations (use '' to hide the text)
    tick_labels = ['', '', '', '', '', '1']

    # 3. Apply them to the Z-axis
    ax.set_zticks(tick_locations)
    ax.set_zticklabels(tick_labels)
    
    ax.zaxis.label.set_rotation(90)

    # Remove background pane fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["linestyle"] = "-"
        axis._axinfo["grid"]["linewidth"] = 0.4
        axis._axinfo["grid"]["color"] = (0.8, 0.8, 0.8, 1.0)

    if outpath:
        fig.savefig(outpath)
        print(f"Saved → {outpath}")
    else:
        plt.show()

    plt.close(fig)


# ── standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from viz_style import apply_thesis_style
    colors = apply_thesis_style()

    result = simulate_dk_moll()
    plot_dk_fdelta(result, colors=colors)
    plot_dk_theta(result, colors=colors)
