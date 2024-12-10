import numpy as np
from grid_methods.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)


def test_D1():

    N = 4
    x_min = -2.0
    x_max = 2.0
    L = x_max - x_min  # Length of the simulation box
    GLL = GaussLegendreLobatto(N, Linear_map(x_min, x_max), symmetrize=False)

    D1 = GLL.D1
    x = GLL.r
    np.testing.assert_allclose(np.dot(D1, x), np.ones(N + 1), atol=1e-14)
    np.testing.assert_allclose(np.dot(D1, 1 - x), -np.ones(N + 1), atol=1e-14)
    np.testing.assert_allclose(np.dot(D1, x**2), 2 * x, atol=1e-14)
    np.testing.assert_allclose(np.dot(D1, 1 - x**2), -2 * x, atol=1e-14)


def test_D2():

    N = 3
    x_min = -1.0
    x_max = 1.0
    L = x_max - x_min  # Length of the simulation box
    GLL = GaussLegendreLobatto(N, Linear_map(x_min, x_max), symmetrize=False)

    G = np.zeros((N + 1, N - 1))
    G[1:N, :] = np.eye(N - 1)

    D1 = GLL.D1
    D2 = GLL.D2
    D1_G = np.dot(D1, G)
    D2_G = np.dot(D1, D1_G)

    print(np.allclose(D1[1:-1, 1:-1], D1_G[1:N, :]))
    print(np.allclose(D2[1:-1, 1:-1], D2_G[1:N, :]))

    print(np.allclose(D2[1:-1, 1:-1], np.dot(D1, D1)[1:-1, 1:-1]))


if __name__ == "__main__":
    test_D2()
