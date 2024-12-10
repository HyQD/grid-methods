import numpy as np
from matplotlib import pyplot as plt
from grid_methods.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)


def test_particle_in_box():
    def test_particle_in_box(L):

        N = 40
        x_min = 0
        x_max = L

        GLL = GaussLegendreLobatto(
            N, Linear_map(x_min, x_max), symmetrize=False
        )
        x = GLL.r[1:-1]
        w = GLL.weights[1:-1]
        x_dot = GLL.r_dot[1:-1]

        D1 = GLL.D1
        D2 = np.dot(D1, D1)[
            1:-1, 1:-1
        ]  # Second derivative matrix with Dirichlet boundary conditions u(x_min) = u(x_max) = 0
        T = -0.5 * D2

        eps, U = np.linalg.eig(T)
        idx = np.argsort(eps)
        eps = eps[idx]
        U = U[:, idx]

        N_max = 10
        for k in range(1, N_max):
            eps_k_approx = eps[k - 1]
            psi_k_approx = U[:, k - 1]
            norm_psi_k_approx = np.dot(w, x_dot * psi_k_approx**2)
            psi_k_approx /= np.sqrt(norm_psi_k_approx)

            eps_k_exact = k**2 * np.pi**2 / (2 * L**2)
            psi_k_exact = np.sqrt(2 / L) * np.sin(k * np.pi * x / L)
            norm_psi_k_exact = np.dot(w, x_dot * psi_k_exact**2)
            np.testing.assert_allclose(
                eps_k_approx, eps_k_exact, rtol=0.0, atol=1e-12
            )
            np.testing.assert_allclose(
                np.abs(psi_k_approx) ** 2,
                np.abs(psi_k_exact) ** 2,
                rtol=0.0,
                atol=1e-12,
            )

    test_particle_in_box(1)
    test_particle_in_box(1.5)
    test_particle_in_box(2)
    test_particle_in_box(4)


def test_harmonic_oscillator():
    from scipy.special import eval_hermite as Hermite, factorial as fac

    def test_harmonic_oscillator(omega):

        N = 80
        a = 0.5 * omega

        """
        np.exp(-a*x**2) < 10**(-16) for |x| > sqrt(-ln(10^(-16))/a), ln(10^-16) \approx -36.84
        """
        x_min = -np.sqrt(37 / a)
        x_max = np.sqrt(37 / a)

        GLL = GaussLegendreLobatto(
            N, Linear_map(x_min, x_max), symmetrize=False
        )
        x = GLL.r[1:-1]
        w = GLL.weights[1:-1]
        x_dot = GLL.r_dot[1:-1]
        D1 = GLL.D1
        D2 = np.dot(D1, D1)[1:-1, 1:-1]
        T = -0.5 * D2
        V = np.diag(0.5 * omega**2 * x**2)
        H = T + V
        eps, U = np.linalg.eig(H)
        idx = np.argsort(eps)
        eps = eps[idx]
        U = U[:, idx]

        for k in range(0, 5):

            eps_k_approx = eps[k]
            eps_k_exact = omega * (k + 0.5)

            psi_k_approx = U[:, k]
            norm_psi_k_approx = np.dot(w, x_dot * psi_k_approx**2)
            psi_k_approx /= np.sqrt(norm_psi_k_approx)

            psi_k_exact = (
                1
                / np.sqrt(2**k * fac(k))
                * (omega / np.pi) ** (0.25)
                * Hermite(k, np.sqrt(omega) * x)
                * np.exp(-0.5 * omega * x**2)
            )
            np.testing.assert_allclose(
                eps_k_approx, eps_k_exact, rtol=0.0, atol=1e-12
            )
            np.testing.assert_allclose(
                np.abs(psi_k_approx) ** 2,
                np.abs(psi_k_exact) ** 2,
                rtol=0.0,
                atol=1e-12,
            )

    test_harmonic_oscillator(1)
    test_harmonic_oscillator(2)
    test_harmonic_oscillator(4)
    test_harmonic_oscillator(0.5)
    test_harmonic_oscillator(0.75)


if __name__ == "__main__":
    test_harmonic_oscillator()
