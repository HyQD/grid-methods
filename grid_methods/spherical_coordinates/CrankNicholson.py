import numpy as np
from utils import get_T_dvr, coeff
import tqdm
from triblockprod import block_tridiag_product
from triblocksolve import block_tridiag_solve
import h5py


def CrankNicholson_sine_DVR(
    r,
    potential,
    l_max,
    time_points,
    laser,
    absorber,
    dump_psi_t=[False, 100, "dat"],
):
    """
    Solve the radial part of the time-dependent Schrödinger for a particle in a spherically symmetric potential interacting with
    a spatially uniform (dipole-approximation) laser pulse/electric field in the length gauge polarized along the z-axis
    with the Crank-Nicholson method.


    Parameters
    ----------
    r :
        The radial grid.
    potential :
        The potential evaluated at the grid points.
    l_max : int
        The highest angular momentum to include in the simulation.
    time_points :

    laser : callable

    absorber : [bool, array]
        If absorber[0] is set to True use an absorber where absorber[1] is the absorbing function evaluated at the grid points.
    dump_psi_t=[bool, int, string]
        If dump_psi_t[0] is set to True dump psi_t to disk every dump_psi_t[1] timesteps.
        dump_psi_t[2] gives the path to where psi_t is to be dumped.

    Returns
    -------
    expec_z : array
        The expectation value of z(time_points).
    psi_t
        The wavefunction at the end of the simulation.
    """

    n_r = len(r)
    dr = r[1] - r[0]
    L = l_max + 1
    num_steps = len(time_points)
    dt = time_points[1] - time_points[0]

    T = get_T_dvr(n_r, dr)

    Identity = np.eye(n_r)
    V_eff_l = np.zeros((L, n_r, n_r))
    R_l = np.zeros((L, n_r, n_r))
    for l in range(L):
        np.fill_diagonal(V_eff_l[l], l * (l + 1) / (2 * r**2) - potential)
        np.fill_diagonal(R_l[l], coeff(l) * r)

    H_0 = T + V_eff_l[0]

    # Diagonalize the l=0 Hamiltonian
    eps, D = np.linalg.eigh(H_0)

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
            expec_z += np.sum(
                np.multiply(r, np.multiply(psi[l].conj(), psi[l + 1]))
            ).real * coeff(l)

        return 2 * expec_z

    psi_t = np.zeros((L, n_r), dtype=np.complex128)

    # The intial state is set to the lowest lying l=0 eigenvector/state.
    psi_t[0] = np.complex128(D[:, 0].copy())

    expec_z = np.zeros(num_steps, dtype=np.complex128)
    expec_z[0] = compute_expec_z(psi_t)

    norm = np.zeros(num_steps)
    norm[0] = np.linalg.norm(psi_t.ravel())

    for n in tqdm.tqdm(range(num_steps - 1)):
        tn = time_points[n]

        lower_p, diagonal_p, upper_p = block_tridiag_Hamiltonian(
            tn + dt / 2, 1j * dt / 2
        )
        lower_m, diagonal_m, upper_m = block_tridiag_Hamiltonian(
            tn + dt / 2, -1j * dt / 2
        )

        z = block_tridiag_product(lower_m, diagonal_m, upper_m, psi_t)
        psi_t = block_tridiag_solve(lower_p, diagonal_p, upper_p, z)

        if absorber[0]:
            for l in range(L):
                psi_t[l] = np.multiply(psi_t[l], absorber[1])

        expec_z[n + 1] = compute_expec_z(psi_t)
        norm[n + 1] = np.linalg.norm(psi_t.ravel())

        if dump_psi_t[0]:
            if (n + 1) % dump_psi_t[1] == 0:
                h5f = h5py.File(f"{dump_psi_t[2]}/psi_step={(n+1)}.h5", "w")
                h5f.create_dataset("psi", data=psi_t, compression="gzip")
                h5f.close()

    return norm, expec_z, psi_t
