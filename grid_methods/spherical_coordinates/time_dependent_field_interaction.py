import numpy as np
from opt_einsum import contract


from grid_methods.spherical_coordinates.rhs import VPsi
from grid_methods.spherical_coordinates.Hpsi_components import (
    x_psi,
    y_psi,
    z_psi,
    x_x_psi,
    y_y_psi,
    px_psi,
    py_psi,
    pz_psi,
    y_px_psi,
)


class V_psi_length_z(VPsi):
    def __init__(
        self, angular_matrix_elements, radial_matrix_elements, e_field_z
    ):
        super().__init__(
            angular_matrix_elements,
            radial_matrix_elements,
            e_field_z=e_field_z,
        )

        self.r = radial_matrix_elements.r
        self.nr = radial_matrix_elements.nr
        self.nl = angular_matrix_elements.n_lm
        self.z_Omega = angular_matrix_elements("z_Omega")

    def __call__(self, psi, t, ravel=True):
        psi_new = self.e_field_z(t) * z_psi(psi, self.z_Omega, self.r)

        if ravel:
            return psi_new.ravel()
        else:
            return psi_new


class V_psi_length(VPsi):
    def __init__(
        self,
        angular_matrix_elements,
        radial_matrix_elements,
        e_field_x=None,
        e_field_y=None,
        e_field_z=None,
    ):
        super().__init__(
            angular_matrix_elements,
            radial_matrix_elements,
            e_field_x=e_field_x,
            e_field_y=e_field_y,
            e_field_z=e_field_z,
        )

        self.x_active = False if e_field_x is None else True
        self.y_active = False if e_field_y is None else True
        self.z_active = False if e_field_z is None else True

        self.r = self.radial_matrix_elements.r

        if self.x_active:
            self.x_Omega = self.angular_matrix_elements("x_Omega")
        if self.y_active:
            self.y_Omega = self.angular_matrix_elements("y_Omega")
        if self.z_active:
            self.z_Omega = self.angular_matrix_elements("z_Omega")

        self.active = True

    def __call__(self, psi, t, ravel=True):
        psi = psi.reshape((self.n_lm, self.nr))
        psi_new = np.zeros((self.n_lm, self.nr), dtype=np.complex128)

        if self.x_active:
            psi_new += self.e_field_x(t) * x_psi(psi, self.x_Omega, self.r)

        if self.y_active:
            psi_new += self.e_field_y(t) * y_psi(psi, self.y_Omega, self.r)

        if self.z_active:
            psi_new += self.e_field_z(t) * z_psi(psi, self.z_Omega, self.r)

        if ravel:
            return psi_new.ravel()
        else:
            return psi_new


class V_psi_velocity_z(VPsi):
    def __init__(
        self,
        angular_matrix_elements,
        radial_matrix_elements,
        a_field_z,
    ):
        super().__init__(
            angular_matrix_elements,
            radial_matrix_elements,
            a_field_z=a_field_z,
        )

        self.r = radial_matrix_elements.r
        self.nr = radial_matrix_elements.nr
        self.nl = angular_matrix_elements.n_lm

        self.r_inv = self.radial_matrix_elements.r_inv
        self.D1 = self.radial_matrix_elements.D1

        self.z_Omega = self.angular_matrix_elements("z_Omega")
        self.H_z_beta = self.angular_matrix_elements("H_z_beta")

    def __call__(self, psi, t, ravel=True):

        dpsi_dr = contract("ij, Ij->Ii", self.D1, psi)

        psi_new = self.a_field_z(t) * pz_psi(
            psi, dpsi_dr, self.z_Omega, self.H_z_beta, self.r_inv
        )

        if ravel:
            return psi_new.ravel()
        else:
            return psi_new


class V_psi_velocity(VPsi):
    def __init__(
        self,
        angular_matrix_elements,
        radial_matrix_elements,
        a_field_x=None,
        a_field_y=None,
        a_field_z=None,
    ):
        super().__init__(
            angular_matrix_elements,
            radial_matrix_elements,
            a_field_x=a_field_x,
            a_field_y=a_field_y,
            a_field_z=a_field_z,
        )

        self.r_inv = self.radial_matrix_elements.r_inv
        self.D1 = self.radial_matrix_elements.D1

        self.x_active = False if a_field_x is None else True
        self.y_active = False if a_field_y is None else True
        self.z_active = False if a_field_z is None else True

        if self.x_active:
            self.x_Omega = self.angular_matrix_elements("x_Omega")
            self.H_x_beta = self.angular_matrix_elements("H_x_beta")
        if self.y_active:
            self.y_Omega = self.angular_matrix_elements("y_Omega")
            self.H_y_beta = self.angular_matrix_elements("H_y_beta")
        if self.z_active:
            self.z_Omega = self.angular_matrix_elements("z_Omega")
            self.H_z_beta = self.angular_matrix_elements("H_z_beta")

        self.active = True

    def __call__(self, psi, t, ravel=True):
        psi = psi.reshape((self.n_lm, self.nr))
        psi_new = np.zeros((self.n_lm, self.nr), dtype=np.complex128)

        dpsi_dr = contract("ij, Ij->Ii", self.D1, psi)

        if self.x_active:
            psi_new += self.a_field_x(t) * px_psi(
                psi, dpsi_dr, self.x_Omega, self.H_x_beta, self.r_inv
            )

        if self.y_active:
            psi_new += self.a_field_y(t) * py_psi(
                psi, dpsi_dr, self.y_Omega, self.H_y_beta, self.r_inv
            )

        if self.z_active:
            psi_new += self.a_field_z(t) * pz_psi(
                psi, dpsi_dr, self.z_Omega, self.H_z_beta, self.r_inv
            )

        if ravel:
            return psi_new.ravel()
        else:
            return psi_new


class V_psi_velocity_first(VPsi):
    def __init__(
        self,
        angular_matrix_elements,
        radial_matrix_elements,
        a_field_x=None,
        a_field_y=None,
        a_field_z=None,
        k_x=None,
        k_y=None,
        k_z=None,
    ):
        super().__init__(
            angular_matrix_elements,
            radial_matrix_elements,
            a_field_x=a_field_x,
            a_field_y=a_field_y,
            a_field_z=a_field_z,
        )

        self.k_x = k_x
        self.k_y = k_y
        self.k_z = k_z

        self.r = self.radial_matrix_elements.r
        self.r_inv = self.radial_matrix_elements.r_inv
        self.D1 = self.radial_matrix_elements.D1

        self.x_polarized = False if a_field_x is None else True
        self.y_polarized = False if a_field_y is None else True
        self.z_polarized = False if a_field_z is None else True

        self.x_propagation = False if k_x is None else True
        self.y_propagation = False if k_y is None else True
        self.z_propagation = False if k_z is None else True

        if self.x_polarized and self.y_propagation:
            self.x_Omega = self.angular_matrix_elements("x_Omega")
            self.y_Omega = self.angular_matrix_elements("y_Omega")

            self.y_x_Omega = self.angular_matrix_elements("y_x_Omega")
            self.y_y_Omega = self.angular_matrix_elements("y_y_Omega")

            self.H_x_beta = self.angular_matrix_elements("H_x_beta")
            self.y_px_beta = self.angular_matrix_elements("y_px_beta")

        if self.y_polarized and self.x_propagation:
            self.x_Omega = self.angular_matrix_elements("x_Omega")
            self.y_Omega = self.angular_matrix_elements("y_Omega")
            self.H_y_beta = self.angular_matrix_elements("H_y_beta")
            self.y_x_Omega = self.angular_matrix_elements("y_x_Omega")
            self.x_py_beta = self.angular_matrix_elements("x_py_beta")
            self.x_x_Omega = self.angular_matrix_elements("x_x_Omega")

    def __call__(self, psi, t, ravel=True):

        psi = psi.reshape((self.n_lm, self.nr))
        psi_new = np.zeros((self.n_lm, self.nr), dtype=np.complex128)

        dpsi_dr = contract("ij, Ij->Ii", self.D1, psi)

        if self.x_polarized and self.y_propagation:

            A1_x_t, A2_x_t = self.a_field_x(t)

            psi_new += A1_x_t * px_psi(
                psi, dpsi_dr, self.x_Omega, self.H_x_beta, self.r_inv
            )
            psi_new += (
                A2_x_t
                * self.k_y
                * y_px_psi(psi, dpsi_dr, self.y_x_Omega, self.y_px_beta, self.r)
            )

            psi_new += 0.5 * A1_x_t**2 * psi
            psi_new += (
                A1_x_t * A2_x_t * self.k_y * y_psi(psi, self.y_Omega, self.r)
            )
            psi_new += (
                0.5
                * A2_x_t**2
                * self.k_y**2
                * y_y_psi(psi, self.y_y_Omega, self.r)
            )

        if self.y_polarized and self.x_propagation:

            A1_y_t, A2_y_t = self.a_field_y(t)

            psi_new += A1_y_t * px_psi(
                psi, dpsi_dr, self.y_Omega, self.H_y_beta, self.r_inv
            )
            psi_new += (
                A2_y_t
                * self.k_x
                * y_px_psi(psi, dpsi_dr, self.y_x_Omega, self.x_py_beta, self.r)
            )

            psi_new += 0.5 * A1_y_t**2 * psi
            psi_new += (
                A1_y_t * A2_y_t * self.k_x * y_psi(psi, self.x_Omega, self.r)
            )
            psi_new += (
                0.5
                * A2_y_t**2
                * self.k_x**2
                * y_y_psi(psi, self.x_x_Omega, self.r)
            )

        if ravel:
            return psi_new.ravel()
        else:
            return psi_new
