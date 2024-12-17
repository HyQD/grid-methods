import numpy as np


def compute_ground_state(
    angular_matrix_elements, radial_matrix_elements, potential, l=0, hermitian=True
):
    nr = radial_matrix_elements.nr
    r = radial_matrix_elements.r
    T_D2 = -(1 / 2) * radial_matrix_elements.D2

    H0 = np.zeros((nr, nr))
    T = T_D2 + np.diag(l * (l + 1) / (2 * r**2))
    V = np.diag(potential)

    H0 = T + V

    if hermitian:
        assert np.allclose(H0, H0.T)
        eps, phi_n = np.linalg.eigh(H0)
    else:
        eps, phi_n = np.linalg.eig(H0)

    return eps, phi_n


def compute_ground_state_diatomic(
    angular_matrix_elements, radial_matrix_elements, potential, l_max, hermitian=True
):
    nl = l_max + 1
    nr = radial_matrix_elements.nr
    r = radial_matrix_elements.r
    T_D2 = -(1 / 2) * radial_matrix_elements.D2

    clmb_ = angular_matrix_elements("1/(r-a)")
    clmb = np.zeros((nl, nl, nr, nr), dtype=np.complex128)
    clmb[:, :, np.arange(nr), np.arange(nr)] = clmb_
    clmb = clmb.swapaxes(1, 2).swapaxes(0, 1).swapaxes(2, 3).T
    clmb = clmb.reshape(nr * nl, nr * nl)

    H0 = np.zeros((nr * nl, nr * nl))
    for l in range(nl):
        T = T_D2 + np.diag(l * (l + 1) / (2 * r**2))
        V = np.diag(potential)

        H0[l * nr : ((l + 1) * nr), l * nr : ((l + 1) * nr)] = T + V

    H0 = H0 - clmb

    if hermitian:
        assert np.allclose(H0, H0.T)
        eps, phi_n = np.linalg.eigh(H0)
    else:
        eps, phi_n = np.linalg.eig(H0)

    return eps, phi_n
