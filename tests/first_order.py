import numpy as np
from matplotlib import pyplot as plt
import tqdm
from opt_einsum import contract
from scipy.sparse.linalg import LinearOperator, bicgstab

from grid_methods.spherical_coordinates.radial_matrix_elements import (
    RadialMatrixElements,
)
from grid_methods.spherical_coordinates.angular_matrix_elements import (
    AngularMatrixElements_lm,
)
from grid_methods.spherical_coordinates.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Rational_map,
    Linear_map,
)

from grid_methods.spherical_coordinates.lasers import (
    square_velocity_dipole,
    square_velocity_first,
)


from grid_methods.spherical_coordinates.utils import mask_function
from grid_methods.spherical_coordinates.Hpsi_components import (
    H0_psi,
)

from grid_methods.spherical_coordinates.preconditioners import M2Psi

from grid_methods.spherical_coordinates.rhs import (
    H0Psi,
    HtPsi,
)

from grid_methods.spherical_coordinates.time_dependent_field_interaction import (
    V_psi_velocity_z,
    V_psi_velocity_first,
)


from grid_methods.spherical_coordinates.utils import (
    Counter,
    quadrature,
)

from grid_methods.spherical_coordinates.properties import expec_x_i, expec_p_i

from grid_methods.spherical_coordinates.ground_state import compute_ground_state

from grid_methods.spherical_coordinates.propagators import BiCGstab


### INPUTS #######################

# pulse inputs
E0 = 0.01
omega = 0.2
ncycles = 1
dt = 0.25
speed_of_light = 137
k_y = omega / speed_of_light

# grid inputs
N = 200
nr = N - 1
r_max = 100
l_max = 3


### SETUP ########################

# setup Legendre-Lobatto grid
gll = GaussLegendreLobatto(N, Linear_map(r_max=r_max))
weights = gll.weights

# setup radial matrix elements
radial_matrix_elements = RadialMatrixElements(gll)
potential = -radial_matrix_elements.r_inv
r = radial_matrix_elements.r
D1 = radial_matrix_elements.D1
T_D2 = -(1 / 2) * radial_matrix_elements.D2

# setup angular matrix elements
angular_matrix_elements = AngularMatrixElements_lm(
    arr_to_calc=[
        "x_Omega",
        "y_Omega",
        "z_Omega",
        "x_x_Omega",
        "y_x_Omega",
        "z_x_Omega",
        "z_y_Omega",
        "y_y_Omega",
        "z_z_Omega",
        "H_x_beta",
        "H_y_beta",
        "H_z_beta",
        "x_py_beta",
        "x_pz_beta",
        "y_px_beta",
        "y_pz_beta",
        "z_px_beta",
        "z_py_beta",
    ],
    l_max=l_max,
)
n_lm = angular_matrix_elements.n_lm

# setup mask function
mask_r = mask_function(r, r[-1], r[-1] - 30)

# Compute ground/intial state
eps, phi_n = compute_ground_state(
    angular_matrix_elements, radial_matrix_elements, potential
)

# setup initial state
psi_t0 = np.zeros((n_lm, nr), dtype=np.complex128)
psi_t0[0] = np.complex128(phi_n[:, 0])
psi_t0[0] /= np.sqrt(quadrature(weights, np.abs(psi_t0[0]) ** 2))
# psi0 = psi_t[0].copy()

# setup pulses
t_cycle = 2 * np.pi / omega
tfinal = ncycles * t_cycle

a_field = square_velocity_first(
    field_strength=E0, omega=omega, ncycles=ncycles, phase=-np.pi / 2
)

# right-hand side
H0_psi = H0Psi(
    angular_matrix_elements,
    radial_matrix_elements,
    potential,
)

Vt_psi_Ex_ky = V_psi_velocity_first(
    angular_matrix_elements,
    radial_matrix_elements,
    a_field_x=a_field,
    k_y=k_y,
)

Vt_psi_Ex_kz = V_psi_velocity_first(
    angular_matrix_elements,
    radial_matrix_elements,
    a_field_x=a_field,
    k_z=k_y,
)

Vt_psi_Ey_kx = V_psi_velocity_first(
    angular_matrix_elements,
    radial_matrix_elements,
    a_field_y=a_field,
    k_x=k_y,
)

Vt_psi_Ey_kz = V_psi_velocity_first(
    angular_matrix_elements,
    radial_matrix_elements,
    a_field_y=a_field,
    k_z=k_y,
)

Vt_psi_Ez_kx = V_psi_velocity_first(
    angular_matrix_elements,
    radial_matrix_elements,
    a_field_z=a_field,
    k_x=k_y,
)

Vt_psi_Ez_ky = V_psi_velocity_first(
    angular_matrix_elements,
    radial_matrix_elements,
    a_field_z=a_field,
    k_y=k_y,
)

rhs_Ex_ky = HtPsi(
    angular_matrix_elements, radial_matrix_elements, H0_psi, [Vt_psi_Ex_ky]
)

rhs_Ex_kz = HtPsi(
    angular_matrix_elements, radial_matrix_elements, H0_psi, [Vt_psi_Ex_kz]
)

rhs_Ey_kx = HtPsi(
    angular_matrix_elements, radial_matrix_elements, H0_psi, [Vt_psi_Ey_kx]
)

rhs_Ey_kz = HtPsi(
    angular_matrix_elements, radial_matrix_elements, H0_psi, [Vt_psi_Ey_kz]
)

rhs_Ez_kx = HtPsi(
    angular_matrix_elements, radial_matrix_elements, H0_psi, [Vt_psi_Ez_kx]
)

rhs_Ez_ky = HtPsi(
    angular_matrix_elements, radial_matrix_elements, H0_psi, [Vt_psi_Ez_ky]
)


propagator = BiCGstab(radial_matrix_elements, angular_matrix_elements)
conv_tol = 1e-8

dat_Ex_ky = propagator.run(rhs_Ex_ky, psi_t0, mask_r, tfinal, dt, conv_tol=conv_tol)
dat_Ex_kz = propagator.run(rhs_Ex_kz, psi_t0, mask_r, tfinal, dt, conv_tol=conv_tol)

expec_x_Ex_ky = dat_Ex_ky["expec_x"]
expec_y_Ex_ky = dat_Ex_ky["expec_y"]
expec_z_Ex_ky = dat_Ex_ky["expec_z"]

expec_x_Ex_kz = dat_Ex_kz["expec_x"]
expec_y_Ex_kz = dat_Ex_kz["expec_y"]
expec_z_Ex_kz = dat_Ex_kz["expec_z"]

assert np.linalg.norm(expec_x_Ex_ky - expec_x_Ex_kz) < conv_tol
assert np.linalg.norm(expec_y_Ex_ky - expec_z_Ex_kz) < conv_tol

dat_Ey_kx = propagator.run(rhs_Ey_kx, psi_t0, mask_r, tfinal, dt, conv_tol=conv_tol)
dat_Ey_kz = propagator.run(rhs_Ey_kz, psi_t0, mask_r, tfinal, dt, conv_tol=conv_tol)

time_points = dat_Ey_kx["time_points"]
expec_x_Ey_kx = dat_Ey_kx["expec_x"]
expec_y_Ey_kx = dat_Ey_kx["expec_y"]
expec_z_Ey_kx = dat_Ey_kx["expec_z"]

expec_x_Ey_kz = dat_Ey_kz["expec_x"]
expec_y_Ey_kz = dat_Ey_kz["expec_y"]
expec_z_Ey_kz = dat_Ey_kz["expec_z"]

assert np.max(np.abs(expec_y_Ey_kx - expec_y_Ey_kz)) < conv_tol
assert np.max(np.abs(expec_x_Ey_kx - expec_z_Ey_kz)) < conv_tol

dat_Ez_kx = propagator.run(rhs_Ez_kx, psi_t0, mask_r, tfinal, dt, conv_tol=conv_tol)
dat_Ez_ky = propagator.run(rhs_Ez_ky, psi_t0, mask_r, tfinal, dt, conv_tol=conv_tol)

time_points = dat_Ez_kx["time_points"]
expec_x_Ez_kx = dat_Ez_kx["expec_x"]
expec_y_Ez_kx = dat_Ez_kx["expec_y"]
expec_z_Ez_kx = dat_Ez_kx["expec_z"]

expec_x_Ez_ky = dat_Ez_ky["expec_x"]
expec_y_Ez_ky = dat_Ez_ky["expec_y"]
expec_z_Ez_ky = dat_Ez_ky["expec_z"]

assert np.max(np.abs(expec_z_Ez_kx - expec_z_Ez_ky)) < conv_tol
assert np.max(np.abs(expec_x_Ez_kx - expec_y_Ez_ky)) < conv_tol
