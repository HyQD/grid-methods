import numpy as np
from matplotlib import pyplot as plt
from lasers import sine_square_laser, linear_laser
from scipy.integrate import simps
from numba import jit
import time
from utils import (
    compute_numerical_states,
    setup_Hamiltonian,
    compute_dipole_moment,
    compute_overlap,
    tridiag_prod,
    TDMAsolver,
)
import tqdm

r_max = 200
dr = 0.25
n_grid = int(r_max / dr)
r = np.arange(1, n_grid + 1) * dr

l_max = 3 + 1
M = l_max * n_grid

H0, Hint = setup_Hamiltonian(r_max, dr, l_max)

psi = np.zeros((l_max, n_grid), dtype=np.complex128)

# Compute numerical groundstate and normalize in the sense that
#   int |u(r)|^2 dr = 1 -> int |R(r)|^2 r^2 dr = 1
# Using the exact groundstate
#   u_10(r) = 2*exp(-r)*r,
# as initial state leads to spurious overlaps and expectation values in time since the wavefunction is not
# consistently normalized with respect to the grid parameters.

# Compute states
n_max = 10
_, eigenstates = compute_numerical_states(l_max, n_max, r)

# Create mapping i -> (l, n) (n is not quantum number)
mapping = {}
i = 0
for l in range(l_max):
    for n in range(n_max):
        mapping[i] = (l, n)
        i += 1

psi[0] = np.complex128(eigenstates[0][:, 0])  # np.complex128(2*np.exp(-r)*r)

size = l_max * n_grid

E0 = 0.05
omega = 0.057

t_cycle = 2 * np.pi / omega
td = 3 * t_cycle


laser = sine_square_laser(E0=E0, omega=omega, td=td)

tfinal = td
dt = 0.65

num_steps = int(tfinal / dt) + 1
print(f"number of time steps: {num_steps}")

time_points = np.linspace(0, tfinal, num_steps)
dipole_moment = np.zeros(num_steps)
dipole_moment[0] = compute_dipole_moment(r, psi)

overlaps = np.zeros((num_steps, n_max * l_max), dtype=np.complex128)
for j in range(n_max * l_max):
    l_j, n_j = mapping[j]
    u_nj_lj = eigenstates[l_j][:, n_j]
    overlaps[0, j] = compute_overlap(r, psi[l_j], u_nj_lj)


Identity_vec = np.ones(M, dtype=np.complex128)

for i in tqdm.tqdm(range(num_steps - 1)):
    ti = i * dt

    z1 = tridiag_prod(
        Identity_vec + 1j * dt / 2 * np.diag(H0),
        1j * dt / 2 * np.diag(H0, k=1),
        1j * dt / 2 * np.diag(H0, k=-1),
        psi.ravel(),
    )
    z1 = z1.reshape(l_max, n_grid).T.ravel()

    z2 = tridiag_prod(
        Identity_vec,
        1j * dt / 2 * laser(ti + dt / 2) * np.diag(Hint, k=1),
        1j * dt / 2 * laser(ti + dt / 2) * np.diag(Hint, k=-1),
        z1,
    )
    z2 = z2.reshape(n_grid, l_max).T.ravel()

    z3 = TDMAsolver(
        -1j * dt / 2 * np.diag(H0, k=-1),
        Identity_vec - 1j * dt / 2 * np.diag(H0),
        -1j * dt / 2 * np.diag(H0, k=1),
        z2,
        size,
    )
    z3 = z3.reshape(l_max, n_grid).T.ravel()

    z4 = TDMAsolver(
        -1j * dt / 2 * laser(ti + dt / 2) * np.diag(Hint, k=-1),
        Identity_vec,
        -1j * dt / 2 * laser(ti + dt / 2) * np.diag(Hint, k=1),
        z3,
        size,
    )

    psi = z4.reshape(n_grid, l_max).T

    dipole_moment[i + 1] = compute_dipole_moment(r, psi)
    for j in range(n_max * l_max):
        l_j, n_j = mapping[j]
        u_nj_lj = eigenstates[l_j][:, n_j]
        overlaps[i + 1, j] = compute_overlap(r, psi[l_j], u_nj_lj)


np.save("time-points", time_points)
np.save("fdm-dip-mom", dipole_moment)

plt.figure()
plt.plot(time_points, -dipole_moment)

plt.figure()
plt.plot(time_points, np.abs(overlaps) ** 2)

sum_ovlp_sq = np.sum(np.abs(overlaps) ** 2, axis=1)
plt.figure()
plt.plot(time_points, sum_ovlp_sq)
plt.axhline(0.98)

plt.show()
