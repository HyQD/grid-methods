############################################################################
import numpy as np
from matplotlib import pyplot as plt
from opt_einsum import contract
from grid_methods.spherical_coordinates.lasers import (
    sine_square_laser
)
import time
import tqdm
from grid_methods.cartesian_coordinates.propagators import RungeKutta4
from grid_methods.cartesian_coordinates.rhs import F_psi, VW_psi
from grid_methods.cartesian_coordinates.potentials import Coulomb
from grid_methods.cartesian_coordinates.sinc_dvr import T_sinc_dvr, compute_w12
from grid_methods.cartesian_coordinates.ground_state import scf_diagonalization

############################################################################
"""
Setup grid and compute initial state with self-consistent field iterations.
"""
L = 10
n_dvr = 128
x = np.linspace(-L, L, n_dvr)
dx = x[1] - x[0]
print(f"Number of DVR points: {n_dvr}, dx: {dx}")
############################################################################
"""
Define time propagation parameters.
"""
F0 = 1e-2
omega = 0.057
t_c = 2 * np.pi / omega
n_cycles = 1

td = n_cycles * t_c
tfinal = n_cycles * t_c + t_c
e_field = sine_square_laser(E0=F0, omega=omega, td=td)

dt = 0.01
############################################################################
#Setup up the Hamiltonian and perform SCF iterations.
T = T_sinc_dvr(n_dvr, dx)
V = Coulomb(x, Z=2, a=1.0)
H = T + np.diag(V)
w12 = compute_w12(x)
eps, e_rhf, C = scf_diagonalization(H, w12, n_docc=1, conv_tol_grad=1e-8, max_iters=100)
print(f"Orbital energy: {eps[0]}, Hartree-Fock energy: {e_rhf}")
############################################################################
"""
Set initial state.
"""
psi0 = np.complex128(C[:, 0])
############################################################################
def run_rk4(psi, t_final, dt, x):

    num_steps = int(tfinal / dt) + 1

    time_points = np.zeros(num_steps)
    expec_x = np.zeros(num_steps, dtype=np.complex128)
    norm = np.zeros(num_steps, dtype=np.complex128)
    norm[0] = np.vdot(psi, psi)

    rhs = F_psi(H, w12, x, e_field)

    rk4 = RungeKutta4(rhs, dt)
    psi_t = psi.copy()

    for n in tqdm.tqdm(range(num_steps - 1)):

        time_points[n + 1] = (n + 1) * dt
        tn = time_points[n]
        psi_t = rk4.step(psi_t, tn)

        rho_t = np.abs(psi_t) ** 2
        expec_x[n + 1] = np.sum(rho_t * x)
        norm[n + 1] = np.vdot(psi_t, psi_t)

    dat = dict()
    dat["time_points"] = time_points
    dat["expec_x"] = expec_x
    dat["norm"] = norm
    dat["x"] = x
    dat["psi"] = psi_t

    return dat


dat_rk4 = run_rk4(psi0, tfinal, dt, x)

plt.figure()
plt.plot(dat_rk4["time_points"], dat_rk4["expec_x"].real)
plt.show()


