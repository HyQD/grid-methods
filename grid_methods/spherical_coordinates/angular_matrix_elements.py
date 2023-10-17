import numpy as np
from numpy.polynomial import legendre
from scipy.special import eval_legendre as Legendre
import time
from scipy.special import sph_harm
from pathlib import Path


class AngularMatrixElements:
    def __init__(self, l_max=5, N=101):
        current_file_location = Path(__file__).parent
        coord = np.loadtxt(
            current_file_location / ("Lebedev/lebedev_%03d.txt" % N)
        )

        self.theta = coord[:, 1] * np.pi / 180
        self.phi = coord[:, 0] * np.pi / 180 + np.pi
        self.weights = coord[:, 2]

        self.sin_th = np.sin(self.theta)
        self.cos_th = np.cos(self.theta)
        self.sin_ph = np.sin(self.phi)
        self.cos_ph = np.cos(self.phi)
        self.exp_m1j_p = np.exp(-1j * self.phi)
        self.exp_p1j_p = np.exp(1j * self.phi)

        self.arr = {}

        self.n_l = l_max + 1

    def kron_delta(self, i, j):
        return int(i == j)

    def a_lm(self, l, m):
        if l >= 0 and abs(m) <= l:
            return np.sqrt(
                ((l + 1) ** 2 - m**2) / ((2 * l + 1) * (2 * l + 3))
            )
        else:
            return 0

    def b_lm(self, l, m):
        if l >= 0 and abs(m) <= l:
            return np.sqrt(
                ((l + m + 1) * (l + m + 2)) / ((2 * l + 1) * (2 * l + 3))
            )
        else:
            return 0

    def c_lm(self, l, m):
        if l >= 0 and abs(m) <= l:
            return np.sqrt(l * (l + 1) - m * (m + 1))
        else:
            return 0

    def l1m1_costh_l2m2(self, l1, m1, l2, m2):
        return (
            self.a_lm(l2, m2) * self.kron_delta(l1, l2 + 1)
            + self.a_lm(l2 - 1, m2) * self.kron_delta(l1, l2 - 1)
        ) * self.kron_delta(m1, m2)

    def l1m1_sinth_ddtheta_l2m2(self, l1, m1, l2, m2):
        return (
            l2 * self.a_lm(l2, m2) * self.kron_delta(l1, l2 + 1)
            - (l2 + 1) * self.a_lm(l2 - 1, m2) * self.kron_delta(l1, l2 - 1)
        ) * self.kron_delta(m1, m2)

    def l1m1_sinth_cosph_l2m2(self, l1, m1, l2, m2):
        integral = (
            self.b_lm(l2 - 1, -m2 - 1)
            * self.kron_delta(l1, l2 - 1)
            * self.kron_delta(m1, m2 + 1)
        )
        integral -= (
            self.b_lm(l2, m2)
            * self.kron_delta(l1, l2 + 1)
            * self.kron_delta(m1, m2 + 1)
        )
        integral -= (
            self.b_lm(l2 - 1, m2 - 1)
            * self.kron_delta(l1, l2 - 1)
            * self.kron_delta(m1, m2 - 1)
        )
        integral += (
            self.b_lm(l2, -m2)
            * self.kron_delta(l1, l2 + 1)
            * self.kron_delta(m1, m2 - 1)
        )
        return 0.5 * integral

    def l1m1_sinth_sinph_l2m2(self, l1, m1, l2, m2):
        integral = (
            self.b_lm(l2 - 1, -m2 - 1)
            * self.kron_delta(l1, l2 - 1)
            * self.kron_delta(m1, m2 + 1)
        )
        integral -= (
            self.b_lm(l2, m2)
            * self.kron_delta(l1, l2 + 1)
            * self.kron_delta(m1, m2 + 1)
        )
        integral += (
            self.b_lm(l2 - 1, m2 - 1)
            * self.kron_delta(l1, l2 - 1)
            * self.kron_delta(m1, m2 - 1)
        )
        integral -= (
            self.b_lm(l2, -m2)
            * self.kron_delta(l1, l2 + 1)
            * self.kron_delta(m1, m2 - 1)
        )
        return -1j * integral / 2

    def l1m1_costh_l2m2_Lebedev(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        integrand = Yl1m1_cc * self.cos_th * Yl2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_sinth_cosph_l2m2_Lebedev(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        integrand = Yl1m1_cc * self.sin_th * self.cos_ph * Yl2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_sinth_sq_cosph_sq_l2m2_Lebedev(
        self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2
    ):
        integrand = Yl1m1_cc * self.sin_th**2 * self.cos_ph**2 * Yl2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_sinph_cosph_sinthsq_l2m2(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        integrand = (
            Yl1m1_cc * self.sin_ph * self.cos_ph * self.sin_th**2 * Yl2m2
        )
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_sinth_sinph_l2m2_Lebedev(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        integrand = Yl1m1_cc * self.sin_th * self.sin_ph * Yl2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_sinth_sq_sinph_sq_l2m2_Lebedev(
        self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2
    ):
        integrand = Yl1m1_cc * self.sin_th**2 * self.sin_ph**2 * Yl2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_sinth_ddtheta_l2m2_Lebedev(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        sinth_ddtheta_l2m2 = (
            m2 * self.cos_th * sph_harm(m2, l2, self.phi, self.theta)
        )
        if np.abs(m2 + 1) <= l2:
            sinth_ddtheta_l2m2 += (
                np.sqrt((l2 - m2) * (l2 + m2 + 1))
                * self.sin_th
                * self.exp_m1j_p
                * sph_harm(m2 + 1, l2, self.phi, self.theta)
            )

        integrand = Yl1m1_cc * sinth_ddtheta_l2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_sinth_sq_l2m2_Lebedev(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        integrand = Yl1m1_cc * self.sin_th**2 * Yl2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_px_l2m2_Lebedev(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        integral = self.c_lm(l2, m2) * self.l1m1_costh_l2m2(
            l1, m1, l2, m2 + 1
        ) - self.c_lm(l2, m2 - 1) * self.l1m1_costh_l2m2(l1, m1, l2, m2 - 1)

        integrand = (
            Yl1m1_cc * self.sin_th * (self.exp_m1j_p - self.exp_p1j_p) * Yl2m2
        )
        integral += m2 * np.sum(4 * np.pi * self.weights * integrand)
        return 0.5 * integral

    def l1m1_y_px_l2m2(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        ###############################################################################
        """
        \left(\cos \phi \cos \theta \frac{\partial}{\partial \theta} - \frac{\sin \phi}{\sin \theta} \frac{\partial}{\partial \phi}\right) Y_{l,m}(\theta, \phi)
        = \frac{1}{2} \left[\cos\theta \left( c_{l,m}Y_{l,m+1} - c_{l,m-1}Y_{l,m-1} \right) + m \sin \theta (e^{-i \phi}-e^{i\phi}) Y_{l,m} \right]
        """
        integrand = (
            0.5 * m2 * self.sin_th * (self.exp_m1j_p - self.exp_p1j_p) * Yl2m2
        )

        if abs(m2 + 1) <= l2:
            integrand += (
                0.5
                * self.c_lm(l2, m2)
                * self.cos_th
                * sph_harm(m2 + 1, l2, self.phi, self.theta)
            )

        if abs(m2 - 1) <= l2:
            integrand -= (
                0.5
                * self.c_lm(l2, m2 - 1)
                * self.cos_th
                * sph_harm(m2 - 1, l2, self.phi, self.theta)
            )

        ###############################################################################
        integrand *= Yl1m1_cc * self.sin_th * self.sin_ph

        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_py_l2m2_Lebedev(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        integral = -self.c_lm(l2, m2) * self.l1m1_costh_l2m2(
            l1, m1, l2, m2 + 1
        ) - self.c_lm(l2, m2 - 1) * self.l1m1_costh_l2m2(l1, m1, l2, m2 - 1)

        integrand = (
            Yl1m1_cc * self.sin_th * (self.exp_m1j_p + self.exp_p1j_p) * Yl2m2
        )
        integral += m2 * np.sum(4 * np.pi * self.weights * integrand)
        return 1j * 0.5 * integral

    def l1m1_x_py_l2m2(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        ###############################################################################

        integrand = (
            1j
            * 0.5
            * m2
            * self.sin_th
            * (self.exp_m1j_p + self.exp_p1j_p)
            * Yl2m2
        )

        if abs(m2 + 1) <= l2:
            integrand -= (
                1j
                * 0.5
                * self.c_lm(l2, m2)
                * self.cos_th
                * sph_harm(m2 + 1, l2, self.phi, self.theta)
            )

        if abs(m2 - 1) <= l2:
            integrand -= (
                1j
                * 0.5
                * self.c_lm(l2, m2 - 1)
                * self.cos_th
                * sph_harm(m2 - 1, l2, self.phi, self.theta)
            )

        ###############################################################################
        integrand *= Yl1m1_cc * self.sin_th * self.cos_ph

        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def __call__(self, name):
        return self.arr[name]


class AngularMatrixElements_l(AngularMatrixElements):
    def __init__(self, arr_to_calc=[], l_max=5, N=101):
        super().__init__(l_max, N)

        self.n_lm = l_max + 1
        self.lm_I, self.I_lm = setup_lm_index_mapping_l(l_max)

        for el in arr_to_calc:
            self.arr[el] = np.zeros((self.n_lm, self.n_lm), dtype=np.complex128)

        arr_to_calc_dict = setup_l_arr_to_calc(arr_to_calc)

        if True in arr_to_calc_dict.values():
            self.setup_l_matrix_elements(arr_to_calc_dict)

    def setup_l_matrix_elements(self, arr_to_calc_dict, m=0):
        n_l = self.n_l

        for l1 in range(n_l):
            for l2 in range(n_l):
                if arr_to_calc_dict["z_Omega"]:
                    self.arr["z_Omega"][l1, l2] = self.l1m1_costh_l2m2(
                        l1, m, l2, m
                    )
                if arr_to_calc_dict["H_z_beta"] and arr_to_calc_dict["z_Omega"]:
                    self.arr["H_z_beta"][l1, l2] = (
                        -self.l1m1_sinth_ddtheta_l2m2(l1, m, l2, m)
                        - self.arr["z_Omega"][l1, l2]
                    )
                elif (
                    arr_to_calc_dict["H_z_beta"]
                    and arr_to_calc_dict["z_Omega"] == False
                ):
                    self.arr["H_z_beta"][
                        l1, l2
                    ] = -self.l1m1_sinth_ddtheta_l2m2(
                        l1, m, l2, m
                    ) - self.l1m1_costh_l2m2(
                        l1, m, l2, m
                    )


class AngularMatrixElements_lm(AngularMatrixElements):
    def __init__(
        self, arr_to_calc=[], lmr_arr_to_calc=[], l_max=5, N=101, nr=None
    ):
        super().__init__(l_max, N)

        self.n_lm = (l_max + 1) ** 2
        self.lm_I, self.I_lm = setup_lm_index_mapping_lm(l_max)

        for el in arr_to_calc:
            self.arr[el] = np.zeros((self.n_lm, self.n_lm), dtype=np.complex128)

        arr_to_calc_dict = setup_lm_arr_to_calc(arr_to_calc)

        if True in arr_to_calc_dict.values():
            self.setup_matrix_elements(arr_to_calc_dict)

    def setup_matrix_elements(self, arr_to_calc_dict):
        tic = time.time()
        n_lm = self.n_lm

        l_max = self.n_l - 1

        for l1 in range(l_max + 1):
            for m1 in range(-l1, l1 + 1):
                I = self.I_lm[f"{l1}{m1}"]
                Yl1m1_cc = sph_harm(m1, l1, self.phi, self.theta).conj()
                for l2 in range(l1 - 2, l1 + 3):
                    if l2 >= 0 and l2 <= l_max:
                        for m2 in range(-l2, l2 + 1):
                            J = self.I_lm[f"{l2}{m2}"]
                            Yl2m2 = sph_harm(m2, l2, self.phi, self.theta)

                            if arr_to_calc_dict["x_Omega"]:
                                self.arr["x_Omega"][
                                    I, J
                                ] = self.l1m1_sinth_cosph_l2m2(l1, m1, l2, m2)

                            if arr_to_calc_dict["x_x_Omega"]:
                                self.arr["x_x_Omega"][
                                    I, J
                                ] = self.l1m1_sinth_sq_cosph_sq_l2m2_Lebedev(
                                    Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                )

                            if arr_to_calc_dict["y_Omega"]:
                                self.arr["y_Omega"][
                                    I, J
                                ] = self.l1m1_sinth_sinph_l2m2(l1, m1, l2, m2)

                            if arr_to_calc_dict["y_y_Omega"]:
                                self.arr["y_y_Omega"][
                                    I, J
                                ] = self.l1m1_sinth_sq_sinph_sq_l2m2_Lebedev(
                                    Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                )

                            if arr_to_calc_dict["z_Omega"]:
                                self.arr["z_Omega"][
                                    I, J
                                ] = self.l1m1_costh_l2m2(l1, m1, l2, m2)

                            if arr_to_calc_dict["y_x_Omega"]:
                                self.arr["y_x_Omega"][
                                    I, J
                                ] = self.l1m1_sinph_cosph_sinthsq_l2m2(
                                    Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                )

                            if arr_to_calc_dict["x_py_beta"]:
                                self.arr["x_py_beta"][I, J] = (
                                    self.l1m1_x_py_l2m2(
                                        Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                    )
                                    - self.arr["y_x_Omega"][I, J]
                                )

                            if arr_to_calc_dict["y_px_beta"]:
                                self.arr["y_px_beta"][I, J] = (
                                    self.l1m1_y_px_l2m2(
                                        Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                    )
                                    - self.arr["y_x_Omega"][I, J]
                                )

                            if arr_to_calc_dict["H_x_beta"]:
                                self.arr["H_x_beta"][I, J] = (
                                    self.l1m1_px_l2m2_Lebedev(
                                        Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                    )
                                    - self.arr["x_Omega"][I, J]
                                )

                            if arr_to_calc_dict["H_y_beta"]:
                                self.arr["H_y_beta"][I, J] = (
                                    self.l1m1_py_l2m2_Lebedev(
                                        Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                    )
                                    - self.arr["y_Omega"][I, J]
                                )

                            if arr_to_calc_dict["H_z_beta"]:
                                self.arr["H_z_beta"][I, J] = (
                                    -self.l1m1_sinth_ddtheta_l2m2(
                                        l1, m1, l2, m2
                                    )
                                    - self.arr["z_Omega"][I, J]
                                )

                            if arr_to_calc_dict["H_Bz_Omega"]:
                                self.arr["H_Bz_Omega"][
                                    I, J
                                ] = self.l1m1_sinth_sq_l2m2_Lebedev(
                                    Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                )
        toc = time.time()
        print(f"Time setup angular matrix elements: {toc-tic}")


class AngularMatrixElements_lmr(AngularMatrixElements):
    def __init__(self, arr_to_calc, nr, r, k, phi_k, theta_k, l_max=5, N=101):
        super().__init__(l_max, N)

        self.n_l = l_max + 1
        self.n_lm = (l_max + 1) ** 2
        self.nr = nr
        self.lm_I, self.I_lm = setup_lm_index_mapping_lm(l_max)

        self.r = r
        self.k = k
        self.phi_k = phi_k
        self.theta_k = theta_k

        for el in arr_to_calc:
            self.arr[el] = np.zeros(
                (self.n_lm, self.n_lm, self.nr), dtype=np.complex128
            )

        arr_to_calc_dict = setup_lmr_arr_to_calc(arr_to_calc)

        if True in arr_to_calc_dict.values():
            self.setup_matrix_elements(arr_to_calc_dict)

    def setup_matrix_elements(self, arr_to_calc_dict):
        r = self.r
        nl = self.n_l
        theta = self.theta
        phi = self.phi
        theta_k = self.theta_k
        phi_k = self.phi_k

        for l1 in range(nl - 1):
            for m1 in range(-l1, l1 + 1):
                Y_l1m1 = sph_harm(m1, l1, phi, theta)
                for l2 in range(nl - 1):
                    for m2 in range(-l2, l2 + 1):
                        Y_l2m2 = sph_harm(m2, l2, phi, theta)
                        for L in range(2 * nl):
                            for M in range(-L, L + 1):
                                if arr_to_calc_dict["expph_costh"]:
                                    cond1 = -m1 - M + m2 == 0
                                    cond2 = (np.abs(l1 - L) <= l2 + 1) and (
                                        l2 - 1 <= l1 + L
                                    )
                                    cond3 = ((l1 + L + l2 + 1) % 2) == 0

                                    if cond1 and cond2 and cond3:
                                        Y_LM = sph_harm(M, L, phi, theta)

                                        F_r = f_r(
                                            r, k, L, M, theta_k, phi_k, sign
                                        )
                                        F_W = (
                                            _l1m1_Y_star_cos_theta_l2m2_Lebedev(
                                                l1,
                                                m1,
                                                l2,
                                                m2,
                                                L,
                                                M,
                                                Y_l1m1,
                                                Y_l2m2,
                                                Y_LM,
                                            )
                                        )

                                        index1 = index_mapping[f"{l1},{m1}"]
                                        index2 = index_mapping[f"{l2},{m2}"]

                                        T[index1, index2, :] += F_W * F_r

                                if arr_to_calc_dict["expph_sinth_ddtheta"]:
                                    cond1 = -m1 - M + m2 == 0
                                    cond2 = (np.abs(l1 - L) <= l2 + 1) and (
                                        l2 - 1 <= l1 + L
                                    )
                                    cond3 = ((l1 + L + l2 + 1) % 2) == 0

                                    if cond1 and cond2 and cond3:
                                        F_r = f_r(
                                            r, k, L, M, theta_k, phi_k, sign
                                        )
                                        F_W = l1m1_Y_star_sin_theta_ddtheta_l2m2_Lebedev(
                                            l1, m1, l2, m2, L, M
                                        )

                                        index1 = index_mapping[f"{l1},{m1}"]
                                        index2 = index_mapping[f"{l2},{m2}"]

                                        T[index1, index2, :] += F_W * F_r

        return T


def setup_lm_index_mapping_l(l_max, m=0):
    lm_I = []
    I_lm = dict()
    I = 0
    for l in range(l_max + 1):
        I_lm[f"{l}{m}"] = I
        lm_I.append((l, m))
        I += 1

    return lm_I, I_lm


def setup_lm_index_mapping_lm(l_max):
    lm_I = []
    I_lm = dict()
    I = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            I_lm[f"{l}{m}"] = I
            lm_I.append((l, m))
            I += 1

    return lm_I, I_lm


def setup_l_arr_to_calc(arr_to_calc_list=[]):
    arr_to_calc_dict = {
        "z_Omega": False,
        "H_z_beta": False,
    }

    return set_boolean_values_to_dict(arr_to_calc_dict, arr_to_calc_list)


def setup_lm_arr_to_calc(arr_to_calc_list=[]):
    arr_to_calc_dict = {
        "H_x_beta": False,
        "H_y_beta": False,
        "H_z_beta": False,
        "x_Omega": False,
        "y_Omega": False,
        "z_Omega": False,
        "x_x_Omega": False,
        "y_y_Omega": False,
        "y_x_Omega": False,
        "x_py_beta": False,
        "y_px_beta": False,
        "H_Bz_Omega": False,
    }

    return set_boolean_values_to_dict(arr_to_calc_dict, arr_to_calc_list)


def setup_lmr_arr_to_calc(arr_to_calc_list):
    arr_to_calc_dict = {}

    return set_boolean_values_to_dict(arr_to_calc_dict, arr_to_calc_list)


def set_boolean_values_to_dict(_dict, _list):
    for el in _list:
        if el not in _dict.keys():
            raise ValueError(f"Calculation of {el} array is not implemented")
        else:
            _dict[el] = True

    return _dict
