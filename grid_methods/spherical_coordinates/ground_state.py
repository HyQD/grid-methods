import numpy as np


def compute_ground_state(angular_matrix_elements, radial_matrix_elements, potential):
    nr = radial_matrix_elements.nr
    r = radial_matrix_elements.r
    T_D2 = -(1 / 2) * radial_matrix_elements.D2

    H0 = np.zeros((nr, nr))
    l_init = 0
    T = T_D2 + np.diag(l_init * (l_init + 1) / (2 * r**2))
    V = np.diag(potential)

    H0 = T + V
    assert np.allclose(H0, H0.T)

    eps, phi_n = np.linalg.eigh(H0)

    return eps, phi_n
