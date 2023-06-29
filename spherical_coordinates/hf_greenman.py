import numpy as np
from numpy.polynomial import legendre
from scipy.special import eval_legendre as Legendre
from sympy.physics.quantum.cg import CG, Wigner3j
from matplotlib import pyplot as plt

"""
[1]: DOI:https://doi.org/10.1103/PhysRevA.82.023406
[2]: https://link.springer.com/book/10.1007/978-3-540-68013-0

This program solves the Hartree-Fock equations in spherical coordinates 
for atoms with closed (sub)shells on a Gauss-Legendre-Lobatto grid as described 
in Ref.[1].  
"""


def compute_v0(lj, A_njlj, r):
    """
    Compute the direct poential
    """
    N = len(r) + 1
    v0 = np.zeros(len(r))
    for k1 in range(len(r)):
        max_rk1_r = np.maximum(r[k1], r)
        v0[k1] = (
            2
            / (N * (N + 1))
            * (2 * lj + 1)
            * np.sum(np.abs(A_njlj) ** 2 * np.divide(1, max_rk1_r))
        )
    return v0


def compute_Fx_lp(lp, lj, A_njlj, r):
    """
    Compute the exchange matrix
    """
    N = len(r) + 1
    tmp = np.zeros((len(r), len(r)))

    for L in range(np.abs(lp - lj), lp + lj + 1):
        C_lpLlj_sq = C(lp, L, lj) ** 2
        for k1 in range(len(r)):
            for k2 in range(len(r)):
                tmp[k1, k2] += (
                    2
                    / (N * (N + 1))
                    * C_lpLlj_sq
                    * A_njlj[k1]
                    * A_njlj[k2]
                    * np.minimum(r[k2], r[k1]) ** L
                    / np.maximum(r[k2], r[k1]) ** (L + 1)
                )

    return tmp


def quadrature(weights, r_dot, f):
    return np.sum(weights * r_dot * f)


def kron_delta(i, j):
    if i == j:
        return 1
    else:
        return 0


def C(l1, l2, l3):
    # Eq.[6]
    cg = float(CG(l1, 0, l2, 0, l3, 0).doit())
    return cg


#############################################################################################
# Setup Legendre-Lobatto grid
N = 100
# Get inner nodes
c = np.zeros((N + 1,))
c[-1] = 1
dc = legendre.legder(c)

nodes = np.zeros(N + 1)
nodes[0] = -1
nodes[-1] = 1
nodes[1:-1] = legendre.legroots(dc)


ddg_tilde = np.zeros((N - 1, N - 1))
inner_nodes = nodes[1:-1]
PN_x = legendre.legval(inner_nodes, c)

# Eq.[17]
for i in range(N - 1):
    for j in range(N - 1):
        if i == j:
            ddg_tilde[i, j] = -1 / 3 * N * (N + 1) / (1 - inner_nodes[i] ** 2)
        else:
            ddg_tilde[i, j] = -2 / (inner_nodes[i] - inner_nodes[j]) ** 2

# Eq.[24]
weights = 2 / (N * (N + 1) * PN_x**2)

# Eq.[25]
r = np.zeros(N - 1)
L_scale = 1.5
r = L_scale * (1 + inner_nodes) / (1 - inner_nodes)

r_dot = np.zeros(N - 1)
r_dot = 2 * L_scale / (1 - inner_nodes) ** 2
#############################################################################################

#############################################################################################

"""
Define shell structure (electron-configuration)
   
   He: Z=2,  l_max = 0, Nj = [0]         , Lj = [0], 
   Be: Z=4,  l_max = 0, Nj = [0, 1]      , Lj = [0, 0]
   Ne: Z=10, l_max = 1, Nj = [0, 1, 0]   , Lj = [0, 0, 1]
   Mg: Z=12, l_max = 1, Nj = [0, 1, 2, 0], Lj = [0, 0, 0, 1]
   Ar: Z=18, l_max = 1, Nj = [0, 1, 2, 0, 1], Lj = [0, 0, 0, 1, 1]
   Ca: Z=20, l_max = 1, Nj = [0, 1, 2, 3, 0, 1], Lj = [0, 0, 0, 0, 1, 1]
   Zn: Z=30, l_max = 2, Nj = [0, 1, 2, 3, 0, 1, 0], Lj = [0, 0, 0, 0, 1, 1, 2]
   Kr: Z=36, l_max = 2, Nj = [0, 1, 2, 3, 0, 1, 2, 0], Lj = [0, 0, 0, 0, 1, 1, 1, 2]
   
"""
Nj = [0, 1, 0]
Lj = [0, 0, 1]

# Setup core Hamiltonian
l_max = 1
Z = 10
H_core = np.zeros((l_max + 1, N - 1, N - 1))
for l in range(l_max + 1):
    for i in range(N - 1):
        for j in range(N - 1):
            H_core[l, i, j] += -0.5 * ddg_tilde[i, j] / (r_dot[i] * r_dot[j])
        H_core[l, i, i] += l * (l + 1) / (2 * r[i] ** 2) - Z / r[i]


A_nl_in = np.zeros((l_max + 1, N - 1, N - 1))
eps_nl = np.zeros((l_max + 1, N - 1))

for l in range(l_max + 1):
    eps_nl[l], A_nl_in[l] = np.linalg.eigh(H_core[l])

for nj, lj in zip(Nj, Lj):
    print(f"{eps_nl[lj][nj]:.6f}")

# Normalize
for nj, lj in zip(Nj, Lj):
    A_nl_in[lj][:, nj] /= np.sqrt(
        np.sum(2 / (N * (N + 1)) * np.abs(A_nl_in[lj][:, nj]) ** 2)
    )
#############################################################################################

#############################################################################################
# Perform SCF iterations
alpha = 0.6  # Linear mixing parameter
print()
for I in range(30):

    v0 = np.zeros(N - 1)
    Fx_lp = np.zeros((l_max + 1, N - 1, N - 1))

    # Compute direct potential
    for nj, lj in zip(Nj, Lj):
        v0 += compute_v0(lj, A_nl_in[lj][:, nj], r)

    # Compute exchange matrix and diagonalize
    A_nl_out = np.zeros((l_max + 1, N - 1, N - 1))
    for lp in range(l_max + 1):
        for nj, lj in zip(Nj, Lj):
            Fx_lp[lp] += compute_Fx_lp(lp, lj, A_nl_in[lj][:, nj], r)
        F_lp = H_core[lp] + 2 * np.diag(v0) - Fx_lp[lp]
        assert np.allclose(F_lp, F_lp.T)
        eps_nl[lp], A_nl_out[lp] = np.linalg.eigh(F_lp)

    # Normalize and apply linear mixing
    for nj, lj in zip(Nj, Lj):
        print(f"{eps_nl[lj][nj]:.6f}")
        A_nl_out[lj][:, nj] /= np.sqrt(
            np.sum(2 / (N * (N + 1)) * np.abs(A_nl_out[lj][:, nj]) ** 2)
        )
        A_nl_in[lj][:, nj] = (1 - alpha) * A_nl_in[lj][
            :, nj
        ] + alpha * A_nl_out[lj][:, nj]
    print()

# Compute <phi_i|r|phi_i> for the occupied orbitals.
for nj, lj in zip(Nj, Lj):
    u_njlj = A_nl_in[lj][:, nj] / np.sqrt(r_dot) * PN_x
    print(quadrature(weights, r_dot, r * np.abs(u_njlj) ** 2))

"""
The orbital energies and expectation values of r for the orbitals are in reasonable 
agreement with the values given in Table 3.3 (chapter 3.4) in the book by Johnson (Ref.[2]).
"""
