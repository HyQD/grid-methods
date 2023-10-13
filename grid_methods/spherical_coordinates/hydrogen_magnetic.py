import numpy as np
from matplotlib import pyplot as plt
from scipy.special import sph_harm

"""
[1]: DOI: 10.1088/0953-4075/41/5/055005
"""

N = 29
coord = np.loadtxt("Lebedev/lebedev_%03d.txt" % N)
theta = coord[:, 1] * np.pi / 180
phi = coord[:, 0] * np.pi / 180 + np.pi
weights = coord[:, 2]


def kron_delta(x1, x2):
    if x1 == x2:
        return 1
    else:
        return 0


def T(k1, k2, dr):
    if k1 == k2:
        return (
            1 / (2 * dr**2) * (-1) ** (k1 - k2) * (np.pi**2 / 3 - 1 / (2 * k1**2))
        )
    else:
        return (
            1
            / (2 * dr**2)
            * (-1) ** (k1 - k2)
            * (2 / (k1 - k2) ** 2 - 2 / (k1 + k2) ** 2)
        )


def Yl1m1_sin2theta_Yl2m2(l1, m1, l2, m2):
    Y_m1_l1 = sph_harm(m1, l1, phi, theta)
    Y_m2_l2 = sph_harm(m2, l2, phi, theta)

    integrand = Y_m1_l1.conj() * np.sin(theta) ** 2 * Y_m2_l2
    integral = np.sum(4 * np.pi * weights * integrand)

    return integral


r_max = 10
dr = 0.15
n_r = int(r_max / dr)
r = np.arange(1, n_r + 1) * dr
B = 1.0
m = 0

l_max = 6
dim = n_r * (l_max + 1 - abs(m))
print(f"Dim={dim}")

H = np.zeros((dim, dim))

row = 0
for l1 in range(abs(m), l_max + 1):
    for k1 in range(n_r):
        col = 0
        for l2 in range(abs(m), l_max + 1):
            for k2 in range(n_r):
                H[row, col] = T(k1 + 1, k2 + 1, dr) * kron_delta(l1, l2)
                H[row, col] += (
                    l2
                    * (l2 + 1)
                    / (2 * r[k2] ** 2)
                    * kron_delta(k1, k2)
                    * kron_delta(l1, l2)
                )
                H[row, col] += -1 / r[k2] * kron_delta(k1, k2) * kron_delta(l1, l2)
                H[row, col] += B * m / 2 * kron_delta(k1, k2) * kron_delta(l1, l2)
                H[row, col] += (
                    B**2
                    / 8
                    * r[k2] ** 2
                    * Yl1m1_sin2theta_Yl2m2(l1, m, l2, m).real
                    * kron_delta(k1, k2)
                )
                col += 1
        row += 1

eps, C = np.linalg.eigh(H)


# Print binding energy as defined in Ref.[1] and groundstate energy.
# The resulting binding energies for m=0,-1,-2 and B=1.0 are in reasonable agreement with
# the values reported in Ref.[1] (Table 1,2, and 3).
print(f"Binding energy: {0.5 * B * (abs(m) + m + 1) - eps[0]}")
print(f"Groundstate energy: {eps[0]}")
