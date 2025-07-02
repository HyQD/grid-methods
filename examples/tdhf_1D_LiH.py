############################################################################
import numpy as np
import tqdm
from matplotlib import pyplot as plt

from grid_lib.spherical_coordinates.lasers import sine_square_laser
from grid_lib.cartesian_coordinates.propagators import (
    Rk4,
    CrankNicolson,
    CMF2,
    Strang,
)
from grid_lib.cartesian_coordinates.rhs import FockOperator
from grid_lib.cartesian_coordinates.potentials import Coulomb, Molecule1D
from grid_lib.cartesian_coordinates.sinc_dvr import (
    T_sinc_dvr,
    compute_w12,
    compute_mean_field,
)
from grid_lib.cartesian_coordinates.ground_state import scf_diagonalization

############################################################################
"""
Setup grid and compute initial state with self-consistent field iterations.
"""
L = 40  # Length of the grid
n_dvr = 2 * (2 * L) + 1  # Number of DVR points
x = np.linspace(-L, L, n_dvr)
dx = x[1] - x[0]  # Grid spacing
print(f"x: [{-L},{L}] a.u., n_dvr: {n_dvr}, dx: {dx}")
############################################################################
# Ref. [1]: 10.1103/PhysRevA.88.023402
# Setup up the 1D-LiH Hamiltonian of Ref.[1] and perform SCF iterations.
n_docc = 2
T = T_sinc_dvr(n_dvr, dx)
V = Molecule1D(R=[-1.15, 1.15], Z=[3, 1], alpha=0.5)
H = T + np.diag(V(x))
w12 = compute_w12(x)
eps, e_rhf, C = scf_diagonalization(
    H, w12, n_docc=n_docc, conv_tol_grad=1e-10, max_iters=500
)
for i in range(n_docc):
    print(f"eps{i}: {eps[i]}")
print(f"RHF energy: {e_rhf}")
############################################################################
"""
Define time propagation parameters.
"""
F0 = 0.0534  # Maximum field strength
omega = 0.06075  # Laser frequency
t_c = 2 * np.pi / omega  # Optical cycle
n_cycles = 3

td = n_cycles * t_c  # Duration of the laser pulse
tfinal = td  # Total time of the simulation
e_field = sine_square_laser(E0=F0, omega=omega, td=td)

dt = 0.1
num_steps = int(tfinal / dt) + 1
print(f"Time step: {dt}, dt/dx^2: {dt/dx**2}, nr. of time steps: {num_steps}")
############################################################################
"""
Set initial state.
"""
psi0 = np.complex128(C)
rho0 = np.sum(np.abs(psi0[:, :n_docc]) ** 2, axis=1)
############################################################################
time_points = np.zeros(num_steps)
orbital_expec_x = np.zeros((num_steps, n_docc), dtype=np.complex128)
expec_x = np.zeros(num_steps)
orbital_norms = np.zeros((num_steps, n_docc), dtype=np.complex128)

for i in range(n_docc):
    expec_x[0] += np.vdot(psi0[:, i], x * psi0[:, i]).real
    orbital_expec_x[0, i] = np.vdot(psi0[:, i], x * psi0[:, i])
    orbital_norms[0, i] = np.vdot(psi0[:, i], psi0[:, i])


integrator = Rk4(H, w12, x, e_field, n_docc, dt, CMF=False)
psi_t = psi0.copy()

for n in tqdm.tqdm(range(num_steps - 1)):

    time_points[n + 1] = (n + 1) * dt
    psi_t = integrator.step(psi_t, time_points[n])

    if hasattr(integrator, "conv_iters"):
        print(f"Number of convergence iterations: {integrator.conv_iters}")

    rho_t = np.sum(
        np.abs(psi_t[:, :n_docc]) ** 2, axis=1
    )  # Time-dependent density: rho(x, t) = \sum_i |psi_i(x, t)|^2
    expec_x[n + 1] = np.sum(rho_t * x)
    for i in range(n_docc):
        orbital_expec_x[n + 1, i] = np.vdot(psi_t[:, i], x * psi_t[:, i])
        orbital_norms[n + 1, i] = np.vdot(psi_t[:, i], psi_t[:, i])

# The computed time-dependent expectation value of x, <x(t)>, is in excellent agreement with Fig.5(a) of Ref.[1].
plt.figure()
plt.plot(
    time_points,
    2 * expec_x.real,
    color="red",
    label=r"$\langle \Phi(t)|x|\Phi(t) \rangle$",
)
plt.ylim(-2.4, -0.6)
plt.legend()

plt.figure()
plt.semilogy(x, rho0, label=r"$\rho(x, t=0)$")
plt.semilogy(x, rho_t, label=r"$\rho(x, t_{final})$")
plt.legend()

plt.figure()
plt.semilogy(
    time_points,
    np.abs(1 - orbital_norms[:, 0].real),
    label=r"$|1-\langle \psi_0(t)|\psi_0(t) \rangle|$",
)
plt.semilogy(
    time_points,
    np.abs(1 - orbital_norms[:, 1].real),
    label=r"$|1-\langle \psi_1(t)|\psi_1(t) \rangle$|",
)
plt.legend()
plt.show()
