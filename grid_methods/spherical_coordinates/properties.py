import numpy as np
from opt_einsum import contract

from grid_methods.spherical_coordinates.utils import quadrature


def expec_x_i(psi, weights, r, xi_Omega):
    """
    Compute
        \braket{\Psi | x_i | \Psi} = \braket{\Psi | r * x_i(\Omega) | \Psi}
    """
    n_lm = psi.shape[0]

    expec_xi = 0 + 0j
    tmp_xi = contract("IJ, Jk->Ik", xi_Omega, psi)
    for I in range(n_lm):
        expec_xi += quadrature(weights, r * psi[I].conj() * tmp_xi[I])
    return expec_xi


def expec_p_i(psi, dpsi_dr, weights, r, xi_Omega, H_xi_beta):

    n_lm = psi.shape[0]
    tmp_alpha = contract("IJ, Jk->Ik", xi_Omega, dpsi_dr)
    tmp_beta = contract("IJ, Jk->Ik", H_xi_beta, psi)

    expec_pi = 0 + 0j

    for I in range(n_lm):
        expec_pi -= 1j * quadrature(weights, psi[I].conj() * tmp_alpha[I])
        expec_pi -= 1j * quadrature(weights, (psi[I].conj() / r) * tmp_beta[I])

    return expec_pi


def compute_norm(psi, weights):
    n_lm = psi.shape[0]
    norm = 0 + 0j
    for I in range(n_lm):
        norm += quadrature(weights, np.abs(psi[I]) ** 2)
    return norm


def compute_ionization_probability(psi, r, radius=20, weights):
    """
    Here the (single) ionization probaility is taken as the probability of finding an electron outside a sphere
    of a given radius.

    P_{ionization}(t) = 1-\int_0^{radius} |\Psi(\mathfbf{r},t)|^2 r^2 dr d\Omega
    """
    n_lm = psi.shape[0]
    r_rad = r[r <= radius]
    ionization_prob = 0
    for I in range(n_lm):
        psi_rad = psi[I][r <= radius]
        ionization_prob += quadrature(weights, np.abs(psi_rad) ** 2)

    return 1 - ionization_prob
