import numpy as np
from numpy.polynomial import legendre
from scipy.special import eval_legendre as Legendre
from sympy.physics.quantum.cg import CG, Wigner3j
from matplotlib import pyplot as plt

"""
[1]: DOI: https://doi.org/10.1007/s10910-020-01144-z
[2]: https://link.springer.com/book/10.1007/978-3-540-68013-0

This program solves the Hartree-Fock equations in spherical coordinates 
for the Neon atom on a Gauss-Legendre-Lobatto grid as described in Ref.[1].

The orbital energies are in exact agreement with those given in Table 3.3 (chapter 3.4)
in Ref.[2].
"""


def coeff(l1, l2, l3):
    # Eq.[6]
    wigner = float(Wigner3j(l1, 0, l2, 0, l3, 0).doit())
    return (2 * l2 + 1) * wigner**2


def kron_delta(i, j):
    if i == j:
        return 1
    else:
        return 0


N = 150

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

Z = 10
l_list = [0, 1]
H_core = np.zeros((N - 1, N - 1, 2))

for l in l_list:
    for i in range(N - 1):
        for j in range(N - 1):
            H_core[i, j, l] += -0.5 * ddg_tilde[i, j] / (r_dot[i] * r_dot[j])
            H_core[i, j, l] += (
                l * (l + 1) / (2 * r[i] ** 2) - Z / r[i]
            ) * kron_delta(i, j)

eps_0, u_n0 = np.linalg.eigh(H_core[:, :, 0])
eps_1, u_n1 = np.linalg.eigh(H_core[:, :, 1])
print(eps_0[0], eps_0[1])
print(eps_1[0], eps_1[1])

A = np.zeros((N - 1, N - 1, 3))
Ainv = np.zeros((N - 1, N - 1, 3))
cal_F = np.zeros((N - 1, N - 1, 3))
Fx = np.zeros((N - 1, N - 1, 2))
F = np.zeros((N - 1, N - 1, 2))


for l2 in range(A.shape[2]):
    A[:, :, l2] = np.einsum(
        "i, ij, j->ij", r / r_dot, ddg_tilde, r / r_dot
    ) - l2 * (l2 + 1) * np.eye(N - 1)
    Ainv[:, :, l2] = np.linalg.inv(A[:, :, l2])
    cal_F[:, :, l2] = np.einsum(
        "i, ij, j->ij",
        PN_x / np.sqrt(r_dot),
        Ainv[:, :, l2],
        PN_x / np.sqrt(r_dot),
    )


# Normalize chi_nl and u_nl according to int chi_nl(r)^2 dr = 1.
u_1s_in = u_n0[:, 0].copy()
chi_1s = PN_x * u_1s_in / np.sqrt(r_dot)
chi_1s /= np.sqrt(np.sum(weights * r_dot * chi_1s**2))
u_1s_in = np.sqrt(r_dot) * chi_1s / PN_x

u_2s_in = u_n0[:, 1].copy()
chi_2s = PN_x * u_2s_in / np.sqrt(r_dot)
chi_2s /= np.sqrt(np.sum(weights * r_dot * chi_2s**2))
u_2s_in = np.sqrt(r_dot) * chi_2s / PN_x

u_2p_in = u_n1[:, 0].copy()
chi_2p = PN_x * u_2p_in / np.sqrt(r_dot)
chi_2p /= np.sqrt(np.sum(weights * r_dot * chi_2p**2))
u_2p_in = np.sqrt(r_dot) * chi_2p / PN_x


alpha = 0.8
print()
for k in range(1, 20):

    # Eq.[38]
    rho_tilde = (
        np.abs(u_1s_in) ** 2 + np.abs(u_2s_in) ** 2 + 3 * np.abs(u_2p_in) ** 2
    )

    # Eq.[36]
    v_H = -np.dot(cal_F[:, :, 0], rho_tilde)

    Fx[:, :, 0] = coeff(0, 0, 0) * (
        np.einsum("i, ij, j->ij", u_1s_in, cal_F[:, :, 0], u_1s_in)
        + np.einsum("i, ij, j->ij", u_2s_in, cal_F[:, :, 0], u_2s_in)
    )

    Fx[:, :, 0] += (
        3
        * coeff(0, 1, 1)
        * np.einsum("i, ij, j->ij", u_2p_in, cal_F[:, :, 1], u_2p_in)
    )

    Fx[:, :, 1] = (
        3
        * coeff(1, 0, 1)
        * (
            np.einsum("i, ij, j->ij", u_1s_in, cal_F[:, :, 1], u_1s_in)
            + np.einsum("i, ij, j->ij", u_2s_in, cal_F[:, :, 1], u_2s_in)
        )
    )

    Fx[:, :, 1] += coeff(1, 1, 0) * np.einsum(
        "i, ij, j->ij", u_2p_in, cal_F[:, :, 0], u_2p_in
    )

    Fx[:, :, 1] += (
        5
        * coeff(1, 1, 2)
        * np.einsum("i, ij, j->ij", u_2p_in, cal_F[:, :, 2], u_2p_in)
    )

    F[:, :, 0] = H_core[:, :, 0] + 2 * np.diag(v_H) + Fx[:, :, 0]
    F[:, :, 1] = H_core[:, :, 1] + 2 * np.diag(v_H) + Fx[:, :, 1]

    eps_0, u_n0 = np.linalg.eigh(F[:, :, 0])
    eps_1, u_n1 = np.linalg.eigh(F[:, :, 1])

    print(f"{eps_0[0]:.6f}, {eps_0[1]:.6f}, {eps_1[0]:.6f}")

    u_1s_out = u_n0[:, 0].copy()
    chi_1s = PN_x * u_1s_out / np.sqrt(r_dot)
    chi_1s /= np.sqrt(np.sum(weights * r_dot * chi_1s**2))
    u_1s_out = np.sqrt(r_dot) * chi_1s / PN_x

    u_2s_out = u_n0[:, 1].copy()
    chi_2s = PN_x * u_2s_out / np.sqrt(r_dot)
    chi_2s /= np.sqrt(np.sum(weights * r_dot * chi_2s**2))
    u_2s_out = np.sqrt(r_dot) * chi_2s / PN_x

    u_2p_out = u_n1[:, 0].copy()
    chi_2p = PN_x * u_2p_out / np.sqrt(r_dot)
    chi_2p /= np.sqrt(np.sum(weights * r_dot * chi_2p**2))
    u_2p_out = np.sqrt(r_dot) * chi_2p / PN_x

    u_1s_in = (1 - alpha) * u_1s_in + alpha * u_1s_out
    u_2s_in = (1 - alpha) * u_2s_in + alpha * u_2s_out
    u_2p_in = (1 - alpha) * u_2p_in + alpha * u_2p_out
