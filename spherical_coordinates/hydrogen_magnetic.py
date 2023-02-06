import numpy as np
from matplotlib import pyplot as plt
from scipy.special import sph_harm
import quadpy
import math

scheme = quadpy.u3.schemes["lebedev_029"]()
theta, phi = scheme.theta_phi
weights = scheme.weights


def kron_delta(x1, x2):
    if x1 == x2:
        return 1
    else:
        return 0


def T(k1, k2, dr):
    if k1 == k2:
        return (
            1
            / (2 * dr**2)
            * (-1) ** (k1 - k2)
            * (np.pi**2 / 3 - 1 / (2 * k1**2))
        )
    else:
        return (
            1
            / (2 * dr**2)
            * (-1) ** (k1 - k2)
            * (2 / (k1 - k2) ** 2 - 2 / (k1 + k2) ** 2)
        )


def Yl1m1_sin2theta_Yl2m2(l1, m1, l2, m2):
    if abs(m1) > l1 or abs(m2) > l2:
        return 0
    else:
        Y_m1_l1 = sph_harm(m1, l1, phi, theta)
        Y_m2_l2 = sph_harm(m2, l2, phi, theta)

        integrand = Y_m1_l1.conj() * np.sin(theta) ** 2 * Y_m2_l2
        integral = np.sum(4 * np.pi * weights * integrand)

        if abs(integral.imag) > 1e-10:
            print(l1, m1, l2, m2)

        return integral


r_max = 10
dr = 0.15
n_r = int(r_max / dr)
r = np.arange(1, n_r + 1) * dr
B = 1.0
m = -1

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
                H[row, col] += (
                    -1 / r[k2] * kron_delta(k1, k2) * kron_delta(l1, l2)
                )
                H[row, col] += (
                    B * m / 2 * kron_delta(k1, k2) * kron_delta(l1, l2)
                )
                H[row, col] += (
                    B**2
                    / 8
                    * r[k2] ** 2
                    * Yl1m1_sin2theta_Yl2m2(l1, m, l2, m).real
                    * kron_delta(k1, k2)
                )
                col += 1
        row += 1

# print(np.allclose(H, H.T))

eps, C = np.linalg.eigh(H)
print(0.5 * B * (abs(m) + m + 1) - eps[0])
print(eps[0])
