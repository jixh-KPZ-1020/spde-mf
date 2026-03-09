import numpy as np

# NOTE: do NOT import matplotlib.pyplot at module level.
# Pure computation helper; no __main__ block (called from mollweide.py or figures.py).


def mollweide_density_surface_data(samples, bins_lon=180, bins_lat=90, eps=1e-12):
    """
    Pure computation: convert sphere samples to Mollweide 3-D surface arrays.

    Parameters
    ----------
    samples : (N, 3) array of unit-sphere points

    Returns
    -------
    dict with keys X, Y, Zsurf  — (bins_lon, bins_lat) arrays for plot_surface
    """
    x, y, z = samples.T
    lon = np.arctan2(y, x)                         # [-pi, pi]
    lat = np.arcsin(np.clip(z, -1.0, 1.0))         # [-pi/2, pi/2]

    H, lon_edges, lat_edges = np.histogram2d(
        lon, lat, bins=[bins_lon, bins_lat], density=True
    )

    lonc = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    latc = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    LON, LAT = np.meshgrid(lonc, latc, indexing="ij")

    # Mollweide implicit auxiliary angle: 2θ + sin(2θ) = π sin(lat)
    TH = np.zeros_like(LAT)
    rhs = np.pi * np.sin(LAT)
    for _ in range(12):  # Newton iterations
        f  = 2 * TH + np.sin(2 * TH) - rhs
        fp = 2 + 2 * np.cos(2 * TH)
        TH = TH - f / (fp + eps)

    X = 2 * np.sqrt(2) / np.pi * LON * np.cos(TH)
    Y = np.sqrt(2) * np.sin(TH)

    # Mask outside the Mollweide ellipse
    mask = (X / (2 * np.sqrt(2)))**2 + (Y / np.sqrt(2))**2 <= 1.0 + 1e-9
    Zsurf = H.copy()
    Zsurf[~mask] = np.nan

    return {"X": X, "Y": Y, "Zsurf": Zsurf}


def plot_mollweide_3d(data, outpath, colors):
    """Render the 3-D Mollweide density surface and save to outpath."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        data["X"], data["Y"], data["Zsurf"],
        rstride=1, cstride=1, linewidth=0, antialiased=True,
    )
    ax.set_xlabel(r"Mollweide $x$")
    ax.set_ylabel(r"Mollweide $y$")
    ax.set_zlabel("density")
    ax.set_title(r"Mollweide density surface: $z = \hat\rho(\lambda,\varphi)$")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
