#!/usr/bin/env python3
"""
2D Anderson Hamiltonian on a torus: H = -Δ + V with periodic BC.

One figure:
  top row:  3D surfaces of log10(|ψ|^2) with shared color scale
  bottom row:  3D surfaces of Re(ψ) (scaled for visibility) with shared z-limits

Dependencies: numpy, scipy, matplotlib
Run: python anderson_3d_eigenfunctions.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.colors as mcolors


def periodic_laplacian_2d(N, L):
    h = L / N
    n = N * N
    A = lil_matrix((n, n), dtype=np.float64)

    def idx(i, j):
        return (i % N) * N + (j % N)

    c = 1.0 / h**2
    for i in range(N):
        for j in range(N):
            p = idx(i, j)
            A[p, p] = 4 * c
            A[p, idx(i + 1, j)] = -c
            A[p, idx(i - 1, j)] = -c
            A[p, idx(i, j + 1)] = -c
            A[p, idx(i, j - 1)] = -c

    return A.tocsr(), h


def iid_potential(N, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((N, N))
    V -= V.mean()
    V /= (V.std() + 1e-12)
    return V


def build_anderson(N, L, disorder, seed):
    Lap, h = periodic_laplacian_2d(N, L)
    V = iid_potential(N, seed)
    H = Lap.tolil()
    H.setdiag(H.diagonal() + disorder * V.reshape(-1))
    return H.tocsr(), h


def compute_lowest(H, k):
    evals, evecs = eigsh(H, k=k, which="SA")
    p = np.argsort(evals)
    return evals[p], evecs[:, p]


def plot_all(evecs, evals, N, L, decades=12, real_scale=200.0, elev=35, azim=-60):
    """
    real_scale multiplies Re(ψ) just for vertical visibility.
    For N~200, real_scale in [50, 500] is typical.
    """
    k = evecs.shape[1]
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    logdens = []
    realparts = []

    for i in range(k):
        psi = evecs[:, i].reshape(N, N)
        dens = (psi * psi).real
        logdens.append(np.log10(dens + 1e-16))
        realparts.append(np.real(psi))

    # Shared scaling for log-density
    dens_vmax = max(Z.max() for Z in logdens)
    dens_vmin = dens_vmax - decades
    dens_norm = mcolors.Normalize(vmin=dens_vmin, vmax=dens_vmax)
    dens_cmap = plt.cm.viridis

    # Shared z-limits for Re(psi)
    realmax = max(np.max(np.abs(R)) for R in realparts)
    zlim = real_scale * realmax

    fig = plt.figure(figsize=(4 * k, 7))

    # Top row: log-density
    for i in range(k):
        ax = fig.add_subplot(2, k, i + 1, projection="3d")
        Z = np.clip(logdens[i], dens_vmin, dens_vmax)
        fc = dens_cmap(dens_norm(Z))
        ax.plot_surface(X, Y, Z, facecolors=fc, linewidth=0, antialiased=True, shade=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_zlim(dens_vmin, dens_vmax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("log10(|ψ|²)")
        ax.set_title(f"log|ψ|²   λ={evals[i]:.4f}")

    # Bottom row: Re(psi), scaled
    for i in range(k):
        ax = fig.add_subplot(2, k, k + i + 1, projection="3d")
        Z = real_scale * realparts[i]
        ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, shade=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_zlim(-zlim, zlim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(f"{real_scale:g}·Re(ψ)")
        ax.set_title("Re(ψ) (scaled)")

    # One shared colorbar for log-density
    sm = plt.cm.ScalarMappable(norm=dens_norm, cmap=dens_cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, shrink=0.6, pad=0.02, label="log10(|ψ|²) (shared scale)")

    plt.tight_layout()
    plt.show()


def main():
    L = 40.0
    N = 200
    disorder = 25.0
    k = 4

    H, h = build_anderson(N, L, disorder, seed=1)
    evals, evecs = compute_lowest(H, k)

    plot_all(evecs, evals, N, L, decades=14, real_scale=300.0)


if __name__ == "__main__":
    main()