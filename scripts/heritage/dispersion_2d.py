import numpy as np
import matplotlib.pyplot as plt

def make_kgrid(N, L):
    dx = L / N
    k1 = 2*np.pi * np.fft.fftfreq(N, d=dx)  # angular wavenumbers
    kx, ky = np.meshgrid(k1, k1, indexing="ij")
    k2 = kx**2 + ky**2
    return kx, ky, k2

def gaussian_wavepacket_2d(N, L, x0, y0, sigma0, k0x=0.0, k0y=0.0):
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    # shortest periodic displacement to center the Gaussian on the torus
    dX = (X - x0 + L/2) % L - L/2
    dY = (Y - y0 + L/2) % L - L/2
    env = np.exp(-(dX**2 + dY**2) / (2*sigma0**2))
    phase = np.exp(1j * (k0x*X + k0y*Y))
    psi0 = env * phase
    psi0 = psi0 / np.linalg.norm(psi0.ravel())
    return psi0

def random_potential(N, W, seed=0, smooth_sigma=0.0, L=1.0):
    """
    Continuum Anderson proxy on the grid:
    V sampled i.i.d. Uniform[-W/2, W/2] at grid points.
    Optional smoothing (Gaussian in Fourier space) to impose correlation length.
    """
    rng = np.random.default_rng(seed)
    V = rng.uniform(-W/2, W/2, size=(N, N))
    if smooth_sigma and smooth_sigma > 0:
        # Gaussian filter in Fourier space with physical length smooth_sigma
        _, _, k2 = make_kgrid(N, L)
        filt = np.exp(-0.5 * (smooth_sigma**2) * k2)
        Vhat = np.fft.fft2(V)
        V = np.real(np.fft.ifft2(Vhat * filt))
        # re-normalize roughly back to comparable amplitude
        V = V / (np.std(V) + 1e-12) * (W/np.sqrt(12))  # std of Uniform[-W/2,W/2] is W/sqrt(12)
    return V

def split_step_propagate(psi0, V, L, dt, nsteps, save_every=10):
    """
    Strang splitting:
      psi <- exp(-i V dt/2) * FFT^{-1}[ exp(-i k^2 dt) * FFT(psi) ] * exp(-i V dt/2)
    for i∂t psi = (-Δ + V) psi with ħ=1 and mass scaled so kinetic is -Δ.
    """
    N = psi0.shape[0]
    _, _, k2 = make_kgrid(N, L)
    kinetic_phase = np.exp(-1j * k2 * dt)  # since (-Δ) -> k^2 in Fourier, exp(-i k^2 dt)

    half_pot = np.exp(-1j * V * dt / 2.0)

    psi = psi0.copy()
    snapshots = []
    times = []

    for step in range(nsteps + 1):
        if step % save_every == 0:
            snapshots.append(psi.copy())
            times.append(step * dt)

        if step == nsteps:
            break

        psi = half_pot * psi
        psi_hat = np.fft.fft2(psi)
        psi_hat *= kinetic_phase
        psi = np.fft.ifft2(psi_hat)
        psi = half_pot * psi

        # keep normalization tight (should be near-unitary already)
        psi = psi / np.linalg.norm(psi.ravel())

    return np.array(times), np.array(snapshots)

def torus_circular_spread(p, L):
    """
    Circular spread per coordinate for a 2D density p on [0,L)^2.
    Returns sigma_x, sigma_y (length units), and PR, IPR.
    """
    N = p.shape[0]
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    p = p / (p.sum() + 1e-15)

    theta_x = 2*np.pi * X / L
    theta_y = 2*np.pi * Y / L

    mx = (p * np.exp(1j*theta_x)).sum()
    my = (p * np.exp(1j*theta_y)).sum()

    Rx = np.abs(mx)
    Ry = np.abs(my)

    # Convert circular concentration into a length-like std estimate
    # sigma = (L/2π) * sqrt(-2 ln R)
    sigx = (L/(2*np.pi)) * np.sqrt(max(0.0, -2.0*np.log(max(Rx, 1e-15))))
    sigy = (L/(2*np.pi)) * np.sqrt(max(0.0, -2.0*np.log(max(Ry, 1e-15))))

    ipr = (p**2).sum()
    pr = 1.0 / (ipr + 1e-15)
    return sigx, sigy, pr, ipr

def run_2d_torus_continuum_anderson(
    N=256, L=200.0, T=80.0, dt=0.02, save_every=50,
    sigma0=6.0, k0x=0.6, k0y=0.2,
    W=2.0, seed=1, smooth_sigma=0.0
):
    nsteps = int(T / dt)

    psi0 = gaussian_wavepacket_2d(N, L, x0=L/2, y0=L/2, sigma0=sigma0, k0x=k0x, k0y=k0y)
    V0 = np.zeros((N, N))
    VA = random_potential(N, W=W, seed=seed, smooth_sigma=smooth_sigma, L=L)

    t, snaps_free = split_step_propagate(psi0, V0, L, dt, nsteps, save_every=save_every)
    _, snaps_and  = split_step_propagate(psi0, VA, L, dt, nsteps, save_every=save_every)

    sig_free = []
    sig_and = []
    pr_free = []
    pr_and = []

    for psi in snaps_free:
        p = np.abs(psi)**2
        sx, sy, pr, _ = torus_circular_spread(p, L)
        sig_free.append(np.sqrt(sx*sx + sy*sy))
        pr_free.append(pr)

    for psi in snaps_and:
        p = np.abs(psi)**2
        sx, sy, pr, _ = torus_circular_spread(p, L)
        sig_and.append(np.sqrt(sx*sx + sy*sy))
        pr_and.append(pr)

    sig_free = np.array(sig_free)
    sig_and = np.array(sig_and)
    pr_free = np.array(pr_free)
    pr_and = np.array(pr_and)

    # Plots: spread and PR
    fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(t, sig_free, label="Free (H = -Δ) on 2D torus")
    ax[0].plot(t, sig_and,  label=f"Anderson (H = -Δ + V), W={W}, smooth={smooth_sigma}")
    ax[0].set_ylabel("Circular spread σ(t)")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(t, pr_free, label="Free PR(t)")
    ax[1].plot(t, pr_and,  label="Anderson PR(t)")
    ax[1].set_xlabel("time t")
    ax[1].set_ylabel("Participation ratio PR(t)")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    # Final densities
    pF = np.abs(snaps_free[-1])**2
    pA = np.abs(snaps_and[-1])**2

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
    im0 = ax[0].imshow(VA.T, origin="lower", aspect="equal")
    ax[0].set_title("Disorder V(x,y)")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(pF.T, origin="lower", aspect="equal")
    ax[1].set_title("Final |ψ|^2 (free)")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(pA.T, origin="lower", aspect="equal")
    ax[2].set_title("Final |ψ|^2 (Anderson)")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_2d_torus_continuum_anderson(
        N=256, L=200.0, T=80.0, dt=0.02, save_every=50,
        sigma0=6.0, k0x=0.7, k0y=0.2,
        W=2.5, seed=2, smooth_sigma=1.5
    )