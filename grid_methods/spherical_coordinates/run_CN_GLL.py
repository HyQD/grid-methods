import numpy as np
import matplotlib.pyplot as plt
from gauss_legendre_lobatto import GaussLegendreLobatto
from scipy.linalg import eig
from scipy.special import erf
from utils import coeff
from lasers import sine_square_laser, sine_laser
import tqdm
from triblockprod import block_tridiag_product
from triblocksolve import block_tridiag_solve
import h5py
import sys

E0 = float(sys.argv[1])  # Maximum field strength/ampltiude
omega = float(sys.argv[2])  # Carrier frequency of the laser pulse
nc = float(sys.argv[3])  # Number of optical cycles: nc * (2pi/omega)
dt = float(sys.argv[4])  # Time step

r_max = float(sys.argv[5])  # Maximum extent of the radial grid
n_r = int(sys.argv[6])  # The number of Legendre-Lobatto grid points

l_max = int(
    sys.argv[7]
)  # The number of angular momenta functions in the time-dependent wavefunction
potential_name = sys.argv[8]  # Name of potential: Couloumb or erfCoulomb
mu = float(sys.argv[9])  # Regularization paramter in erfCoulomb: erf(mu*r)/r
V0 = float(sys.argv[10])  # Depth of Gaussian potential
alpha = float(sys.argv[11])  # Width of Gaussian potential

GLL = GaussLegendreLobatto(0, r_max, n_r)
r = GLL.nodes
r = r[1:-1]
n_r = len(r)

L = l_max + 1

t_cycle = 2 * np.pi / omega
td = nc * t_cycle
laser = sine_square_laser(E0=E0, omega=omega, td=td)
# laser = sine_laser(E0=E0, omega=omega, td=td, phase=np.pi/2)
tfinal = td
sampling_rate = 200
save_psi = False

num_steps = int(np.ceil(tfinal) / dt)
time_points = np.arange(0, num_steps) * dt

if potential_name == "Coulomb":
    potential = -1 / r
elif potential_name == "erfCoulomb":
    potential = -erf(mu * r) / r
    potential_name = f"erfCoulomb_mu={mu}"
elif potential_name == "Gaussian":
    potential = -V0 * np.exp(-alpha * r**2)
else:
    print(f"Undefined potential")
    sys.exit(1)


T = -0.5 * GLL.Dx @ GLL.Dx
T = T[1:-1, 1:-1]

Identity = np.eye(n_r)
V_eff_l = np.zeros((L, n_r, n_r))
R_l = np.zeros((L, n_r, n_r))
for l in range(L):
    np.fill_diagonal(V_eff_l[l], l * (l + 1) / (2 * r**2) + potential)
    np.fill_diagonal(R_l[l], coeff(l) * r)

H_0 = T + V_eff_l[0]
H_1 = T + V_eff_l[1]

eps, D = np.linalg.eig(H_0)
idx = eps.argsort()
eps = eps[idx]
D = D[:, idx]

eps_1, D_1 = np.linalg.eig(H_1)
idx_1 = eps_1.argsort()
eps_1 = eps_1[idx_1]
D_1 = D_1[:, idx_1]

print(eps[0:3])
print(eps_1[0:5])

print()
print(f"** Simulation parameters **")
print(f"E0: {E0}, omega: {omega}, nr. of optical cycles: {nc}, dt: {dt}")
print(f"rmax: {r_max}, nr. of grid points: {n_r+2}, lmax: {l_max}")
print(f"Potential: {potential_name}")
print(f"Ground state energy: {eps[0]}")
print()


def block_tridiag_Hamiltonian(t, idt2):
    diagonal = np.zeros((L, n_r, n_r), dtype=np.complex128)
    upper = np.zeros((L - 1, n_r, n_r), dtype=np.complex128)
    lower = np.zeros((L - 1, n_r, n_r), dtype=np.complex128)

    for l in range(L):
        diagonal[l] = Identity + idt2 * (T + V_eff_l[l])
        if l < L - 1:
            upper[l] = idt2 * laser(t) * R_l[l + 1]
            lower[l] = idt2 * laser(t) * R_l[l]
    return lower, diagonal, upper


def compute_expec_z(psi):
    expec_z = 0

    for l in range(0, L - 1):
        expec_z += GLL.quad(
            np.multiply(r, np.multiply(psi[l].conj(), psi[l + 1]))
        ).real * coeff(l)

    return 2 * expec_z


def compute_norm(psi):
    norm = 0
    for l in range(L):
        norm += GLL.quad(np.abs(psi[l]) ** 2)
    return norm


u_1s = D[:, 0] / np.sqrt(GLL.quad(np.abs(D[:, 0]) ** 2))
u_2p = D_1[:, 0] / np.sqrt(GLL.quad(np.abs(D_1[:, 0]) ** 2))


psi_t = np.zeros((L, n_r), dtype=np.complex128)
psi_t[0] = np.complex128(u_1s.copy())
populations = np.zeros((2, num_steps))

if save_psi:
    h5f = h5py.File(
        f"dat/psi_n={0}_E0={E0}_omega={omega}_nc={nc}_dt={dt}_rmax={r_max}_nr={n_r+2}_lmax={l_max}_{potential_name}.h5",
        "w",
    )
    h5f.create_dataset("psi", data=psi_t, compression="gzip")
    h5f.close()

expec_z = np.zeros(num_steps, dtype=np.complex128)
norm = np.zeros(num_steps)

expec_z[0] = compute_expec_z(psi_t)
norm[0] = compute_norm(psi_t)

populations[0][0] = np.abs(GLL.quad(u_1s * psi_t[0])) ** 2
populations[1][0] = np.abs(GLL.quad(u_2p * psi_t[1])) ** 2


for n in tqdm.tqdm(range(num_steps - 1)):
    tn = time_points[n]

    lower_p, diagonal_p, upper_p = block_tridiag_Hamiltonian(tn + dt / 2, 1j * dt / 2)
    lower_m, diagonal_m, upper_m = block_tridiag_Hamiltonian(tn + dt / 2, -1j * dt / 2)

    z = block_tridiag_product(lower_m, diagonal_m, upper_m, psi_t)
    psi_t = block_tridiag_solve(lower_p, diagonal_p, upper_p, z)

    if save_psi:
        if (n + 1) % sampling_rate == 0:
            h5f = h5py.File(
                f"dat/psi_n={(n+1)}_E0={E0}_omega={omega}_nc={nc}_dt={dt}_rmax={r_max}_nr={n_r+2}_lmax={l_max}_{potential_name}.h5",
                "w",
            )
            h5f.create_dataset("psi", data=psi_t, compression="gzip")
            h5f.close()

    expec_z[n + 1] = compute_expec_z(psi_t)
    norm[n + 1] = compute_norm(psi_t)
    populations[0][n + 1] = np.abs(GLL.quad(u_1s * psi_t[0])) ** 2
    populations[1][n + 1] = np.abs(GLL.quad(u_2p * psi_t[1])) ** 2

if save_psi:
    h5f = h5py.File(
        f"dat/psi_n={num_steps}_E0={E0}_omega={omega}_nc={nc}_dt={dt}_rmax={r_max}_nr={n_r+2}_lmax={l_max}_{potential_name}.h5",
        "w",
    )
    h5f.create_dataset("psi", data=psi_t, compression="gzip")
    h5f.close()

h5f = h5py.File(
    f"dat/output_E0={E0}_omega={omega}_nc={nc}_dt={dt}_rmax={r_max}_nr={n_r+2}_lmax={l_max}_{potential_name}.h5",
    "w",
)
h5f.create_dataset("time_points", data=time_points, compression="gzip")
h5f.create_dataset("expec_z", data=expec_z, compression="gzip")
h5f.create_dataset("norm", data=norm, compression="gzip")
h5f.create_dataset("potential", data=potential, compression="gzip")
h5f.create_dataset("r", data=r, compression="gzip")
h5f.create_dataset("laser", data=laser(time_points), compression="gzip")
h5f.create_dataset("eigenvalues", data=eps, compression="gzip")
h5f.close()

plt.figure()
plt.plot(time_points, expec_z.real)

plt.figure()
plt.plot(
    time_points,
    populations[0],
    label=r"$|\langle \psi_{1s}|\Psi(t) \rangle|^2$",
)
plt.plot(
    time_points,
    populations[1],
    label=r"$|\langle \psi_{2p}|\Psi(t) \rangle|^2$",
)
plt.legend()

plt.figure()
plt.plot(time_points, norm)

plt.show()
