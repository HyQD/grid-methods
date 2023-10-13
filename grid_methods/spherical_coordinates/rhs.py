import numpy as np
from opt_einsum import contract


class H0Psi:
    def __init__(
        self,
        angular_matrix_elements,
        radial_matrix_elements,
        potential,
    ):
        self.n_lm = angular_matrix_elements.n_lm
        self.nr = radial_matrix_elements.nr
        lm_I = angular_matrix_elements.lm_I
        r = radial_matrix_elements.r

        self.potential = potential
        self.T_D2 = -(1 / 2) * radial_matrix_elements.D2

        self.centrifugal_force_r = 1 / (2 * r**2)
        self.centrifugal_force_lm = np.zeros(self.n_lm)
        for I in range(self.n_lm):
            l, m = lm_I[I]
            self.centrifugal_force_lm[I] = l * (l + 1)

    def __call__(self, psi, t, ravel=True):
        psi_new = np.zeros((self.n_lm, self.nr), dtype=np.complex128)

        psi_new += contract("Ij, ij->Ii", psi, self.T_D2)
        psi_new += contract("Ik, k->Ik", psi, self.potential)
        psi_temp = contract("I, Ii->Ii", self.centrifugal_force_lm, psi)
        psi_new += contract("i, Ii->Ii", self.centrifugal_force_r, psi_temp)

        if ravel:
            return psi_new.ravel()
        else:
            return psi_new


class VPsi:
    def __init__(self, angular_matrix_elements, radial_matrix_elements, **kwargs):
        self.angular_matrix_elements = angular_matrix_elements
        self.radial_matrix_elements = radial_matrix_elements
        self.__dict__.update(kwargs)

        self.nr = radial_matrix_elements.nr
        self.n_l = angular_matrix_elements.n_l
        self.n_lm = angular_matrix_elements.n_lm

    def __call__(self, psi, t):
        pass


class HtPsi:
    def __init__(
        self, angular_matrix_elements, radial_matrix_elements, H0_psi, V_psi_list
    ):
        self.H0_psi = H0_psi
        self.V_psi_list = V_psi_list

        self.n_lm = angular_matrix_elements.n_lm
        self.nr = radial_matrix_elements.nr

    def __call__(self, psi, t, ravel=True):
        psi = psi.reshape((self.n_lm, self.nr))

        H_psi_t = self.H0_psi(psi, t, ravel=False)

        for V_psi in self.V_psi_list:
            H_psi_t += V_psi(psi, t, ravel=False)

        if ravel:
            return H_psi_t.ravel()
        else:
            return H_psi_t
