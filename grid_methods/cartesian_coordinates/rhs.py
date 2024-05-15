import numpy as np
from grid_methods.cartesian_coordinates.sinc_dvr import mean_field


class RHS:
    def __init__(self, H, w12, x, e_field):
        self.H = H
        self.x = x
        self.e_field = e_field
        self.w12 = w12

    def __call__(self, psi, t):
        pass


class F_psi(RHS):
    def __call__(self, psi, t):
        rhs = np.dot(self.H, psi)
        rhs += self.e_field(t) * self.x * psi
        rhs += mean_field(self.w12, psi, psi) * psi
        return -1j * rhs


class VW_psi(RHS):
    def __call__(self, psi, t):
        rhs = self.e_field(t) * self.x * psi
        rhs += mean_field(self.w12, psi, psi) * psi
        return -1j * rhs


# def rhs_Wc(psi, t, direct):
#     """
#     Evaluate the right-hand side
#         (H0+V(t)+W(psi))*psi = (H0+V(t)+int w(x,x')|psi(x')|^2 dx')*psi
#     with a constant mean-field.
#     A constant mean-field means that the direct potential is held fixed over some time interval.
#     """
#     rhs = np.dot(H, psi)
#     rhs += e_field(t) * x * psi
#     rhs += direct * psi
#     return -1j * rhs


# def rhs_VWc(psi, t, direct):
#     """
#     Evaluate V(t)*psi+W_{direct}(psi)*psi where the direct potential is fixed/provided.
#     """
#     rhs = e_field(t) * x * psi
#     rhs += direct * psi
#     return -1j * rhs
