#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import expm_multiply


# -------------------------------
# periodic Laplacian
# -------------------------------
def periodic_laplacian(N, L):

    h = L / N
    n = N*N
    A = lil_matrix((n,n))

    def idx(i,j):
        return (i % N)*N + (j % N)

    c = 1/h**2

    for i in range(N):
        for j in range(N):

            p = idx(i,j)

            A[p,p] = 4*c
            A[p,idx(i+1,j)] = -c
            A[p,idx(i-1,j)] = -c
            A[p,idx(i,j+1)] = -c
            A[p,idx(i,j-1)] = -c

    return A.tocsr()


# -------------------------------
# random potential
# -------------------------------
def iid_potential(N, seed=0):

    rng = np.random.default_rng(seed)

    V = rng.standard_normal((N,N))

    V -= V.mean()
    V /= V.std()

    return V


# -------------------------------
# build Hamiltonian
# -------------------------------
def build_anderson(N, L, disorder, seed):

    Lap = periodic_laplacian(N,L)

    V = iid_potential(N,seed)

    H = Lap.tolil()
    H.setdiag(H.diagonal() + disorder*V.reshape(-1))

    return H.tocsr()


# -------------------------------
# delta initial state
# -------------------------------
def initial_state(N):

    psi = np.zeros(N*N, dtype=complex)

    center = N//2
    psi[center*N + center] = 1

    return psi


# -------------------------------
# 3D surface plot
# -------------------------------
def plot_surface(ax, psi, N, L, title):

    dens = np.abs(psi)**2

    Z = np.log10(dens.reshape(N,N)+1e-16)

    x = np.linspace(0,L,N)
    y = np.linspace(0,L,N)

    X,Y = np.meshgrid(x,y,indexing="ij")

    ax.plot_surface(X,Y,Z,cmap="viridis",
                    linewidth=0,
                    antialiased=True)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("log10(|ψ|²)")


# -------------------------------
# main
# -------------------------------
def main():

    N = 80
    L = 20

    disorder_weak = 2
    disorder_strong = 50

    times = [0,1,3,6]

    psi0 = initial_state(N)

    H1 = build_anderson(N,L,disorder_weak,seed=1)
    H2 = build_anderson(N,L,disorder_strong,seed=1)

    A1 = -1j*H1
    A2 = -1j*H2

    fig = plt.figure(figsize=(12,6))

    k = 1

    for t in times:

        psi = expm_multiply(A1,psi0,start=0,stop=t,num=2)[-1]

        ax = fig.add_subplot(2,len(times),k,projection="3d")

        plot_surface(ax,psi,N,L,f"weak W={disorder_weak}, t={t}")

        k+=1

    for t in times:

        psi = expm_multiply(A2,psi0,start=0,stop=t,num=2)[-1]

        ax = fig.add_subplot(2,len(times),k,projection="3d")

        plot_surface(ax,psi,N,L,f"strong W={disorder_strong}, t={t}")

        k+=1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()