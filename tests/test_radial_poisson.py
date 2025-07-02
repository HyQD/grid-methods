import numpy as np
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates.radial_poisson import (
    solve_radial_Poisson_dvr,
)
from matplotlib import pyplot as plt
from scipy.special import erf


def test_radial_poisson():

    """
    The test solves the radial Poisson equation
    .. math::
        \frac{\partial^2}{\partial r^2} \tilde{V}_0(r) = -1/r * u(r)^2,
    for the two test cases:

    1. u(r) = N*r*exp(-r^2), where N is the normalization constant such that
       .. math::
           \int_0^{r_{max}} dr r^2 * u(r)^2 = 1
        The exact solution is given by
        .. math::
            \tilde{V}_0(r) = \text{erf}(\sqrt{2} * r)
    2. u(r) = N*r*exp(-r), where N is the normalization.
        The exact solution is given by
        .. math::
            \tilde{V}_0(r) = 1 - (r + 1) * exp(-2*r)

    The numerical solution (at the inner grid points) is obtained from the solution of the radial Poisson equation in the DVR basis
    .. math::
        \tilde{v}_L(r_\alpha) = \sum_{\beta} u(r_\beta)  \tilde{v}^{DVR}_{L}(r_\alpha; \chi_\beta) u(r_\beta)

    """

    N_r = 128
    r_min = 0
    r_max = 40
    GLL = GaussLegendreLobatto(N_r, Linear_map(r_min, r_max), symmetrize=False)

    tilde_V0_dvr = solve_radial_Poisson_dvr(GLL, n_L=1)

    r = GLL.r[1:-1]
    w_r = GLL.weights[1:-1]
    r_dot = GLL.r_dot[1:-1]

    u_1s_gaussian = r * np.exp(-(r**2))
    norm_psi = np.dot(w_r, r_dot * u_1s_gaussian**2)
    u_1s_gaussian /= np.sqrt(norm_psi)

    tilde_V0 = np.einsum(
        "b, ab, b->a",
        u_1s_gaussian,
        tilde_V0_dvr[0],
        u_1s_gaussian,
        optimize=True,
    )
    tilde_V0_exact_1s_gaussian = erf(
        np.sqrt(2) * r
    )  # The exact solution of the radial Poisson equation to a 1s Gaussian

    assert np.linalg.norm(tilde_V0 - tilde_V0_exact_1s_gaussian) < 1e-10
    assert np.max(np.abs(tilde_V0 - tilde_V0_exact_1s_gaussian)) < 1e-10

    u_1s = r * np.exp(-r)
    norm_1s = np.dot(w_r, r_dot * u_1s**2)
    u_1s /= np.sqrt(norm_1s)
    tilde_V0_1s = np.einsum(
        "b, ab, b->a", u_1s, tilde_V0_dvr[0], u_1s, optimize=True
    )

    tilde_V0_exact_1s = 1 - (r + 1) * np.exp(
        -2 * r
    )  # The exact solution of the radial Poisson equation to a 1s Gaussian

    assert np.linalg.norm(tilde_V0_1s - tilde_V0_exact_1s) < 1e-10
    assert np.max(np.abs(tilde_V0_1s - tilde_V0_exact_1s)) < 1e-10
