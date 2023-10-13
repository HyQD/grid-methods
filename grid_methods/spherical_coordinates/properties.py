from utils import quadrature
from opt_einsum import contract


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


def expec_p_i(psi, dpsi_dr, r, xi_Omega, H_xi_beta):
    tmp_alpha = contract("IJ, Jk->Ik", xi_Omega, dpsi_dr)
    tmp_beta = contract("IJ, Jk->Ik", H_xi_beta, psi)

    expec_pi = 0 + 0j

    for I in range(n_lm):
        expec_pi -= 1j * quadrature(weights, psi[I].conj() * tmp_alpha[I])
        expec_pi -= 1j * quadrature(weights, (psi[I].conj() / r) * tmp_beta[I])

    return expec_pi
