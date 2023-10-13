import numpy as np
from scipy.special import sph_harm

N = 29
coord = np.loadtxt("Lebedev/lebedev_%03d.txt" % N)
theta = coord[:, 1] * np.pi / 180
phi = coord[:, 0] * np.pi / 180 + np.pi
weights = coord[:, 2]


def kron_delta(i, j):
    return int(i == j)


def a_lm(l, m):
    if l >= 0:
        return np.sqrt(((l + 1) ** 2 - m**2) / ((2 * l + 1) * (2 * l + 3)))
    else:
        return 0


def b_lm(l, m):
    if l >= 0:
        return np.sqrt(((l + m + 1) * (l + m + 2)) / ((2 * l + 1) * (2 * l + 3)))
    else:
        return 0


def l1m1_cos_theta_l2m2(l1, m1, l2, m2):
    return (
        a_lm(l2, m2) * kron_delta(l1, l2 + 1)
        + a_lm(l2 - 1, m2) * kron_delta(l1, l2 - 1)
    ) * kron_delta(m1, m2)


def l1m1_cos_theta_l2m2_Lebedev(l1, m1, l2, m2):
    m1_l1 = sph_harm(m1, l1, phi, theta)
    m2_l2 = sph_harm(m2, l2, phi, theta)

    integrand = m1_l1.conj() * np.cos(theta) * m2_l2
    integral = np.sum(4 * np.pi * weights * integrand)

    return integral


def l1m1_sin_theta_cos_phi_l2m2(l1, m1, l2, m2):
    integral = b_lm(l2 - 1, -m2 - 1) * kron_delta(l1, l2 - 1) * kron_delta(m1, m2 + 1)
    integral -= b_lm(l2, m2) * kron_delta(l1, l2 + 1) * kron_delta(m1, m2 + 1)
    integral -= b_lm(l2 - 1, m2 - 1) * kron_delta(l1, l2 - 1) * kron_delta(m1, m2 - 1)
    integral += b_lm(l2, -m2) * kron_delta(l1, l2 + 1) * kron_delta(m1, m2 - 1)
    return 0.5 * integral


def l1m1_sin_theta_cos_phi_l2m2_Lebedev(l1, m1, l2, m2):
    m1_l1 = sph_harm(m1, l1, phi, theta)
    m2_l2 = sph_harm(m2, l2, phi, theta)

    integrand = m1_l1.conj() * np.sin(theta) * np.cos(phi) * m2_l2
    integral = np.sum(4 * np.pi * weights * integrand)

    return integral


def l1m1_sin_theta_sin_phi_l2m2(l1, m1, l2, m2):
    integral = b_lm(l2 - 1, -m2 - 1) * kron_delta(l1, l2 - 1) * kron_delta(m1, m2 + 1)
    integral -= b_lm(l2, m2) * kron_delta(l1, l2 + 1) * kron_delta(m1, m2 + 1)
    integral += b_lm(l2 - 1, m2 - 1) * kron_delta(l1, l2 - 1) * kron_delta(m1, m2 - 1)
    integral -= b_lm(l2, -m2) * kron_delta(l1, l2 + 1) * kron_delta(m1, m2 - 1)
    return -1j * integral / 2


def l1m1_sin_theta_sin_phi_l2m2_Lebedev(l1, m1, l2, m2):
    m1_l1 = sph_harm(m1, l1, phi, theta)
    m2_l2 = sph_harm(m2, l2, phi, theta)

    integrand = m1_l1.conj() * np.sin(theta) * np.sin(phi) * m2_l2
    integral = np.sum(4 * np.pi * weights * integrand)

    return integral


def l1m1_sin_theta_ddtheta_l2m2(l1, m1, l2, m2):
    return (
        l2 * a_lm(l2, m2) * kron_delta(l1, l2 + 1)
        - (l2 + 1) * a_lm(l2 - 1, m2) * kron_delta(l1, l2 - 1)
    ) * kron_delta(m1, m2)


def l1m1_sin_theta_ddtheta_l2m2_Lebedev(l1, m1, l2, m2):
    m1_l1 = sph_harm(m1, l1, phi, theta)

    sin_theta_ddtheta_l2m2 = m2 * np.cos(theta) * sph_harm(m2, l2, phi, theta)
    if np.abs(m2 + 1) <= l2:
        sin_theta_ddtheta_l2m2 += (
            np.sqrt((l2 - m2) * (l2 + m2 + 1))
            * np.sin(theta)
            * np.exp(-1j * phi)
            * sph_harm(m2 + 1, l2, phi, theta)
        )

    integrand = m1_l1.conj() * sin_theta_ddtheta_l2m2
    integral = np.sum(4 * np.pi * weights * integrand)

    return integral


def l1m1_sin_theta_sq_l2m2_Lebedev(l1, m1, l2, m2):
    m1_l1 = sph_harm(m1, l1, phi, theta)
    m2_l2 = sph_harm(m2, l2, phi, theta)

    integrand = m1_l1.conj() * np.sin(theta) ** 2 * m2_l2
    integral = np.sum(4 * np.pi * weights * integrand)

    return integral


l_max = 10
for l1 in range(l_max + 1):
    for m1 in range(-l1, l1 + 1):
        for l2 in range(l_max + 1):
            for m2 in range(-l2, l2 + 1):
                diff = l1m1_cos_theta_l2m2_Lebedev(
                    l1, m1, l2, m2
                ) - l1m1_cos_theta_l2m2(l1, m1, l2, m2)

                diff_2 = l1m1_sin_theta_ddtheta_l2m2_Lebedev(
                    l1, m1, l2, m2
                ) - l1m1_sin_theta_ddtheta_l2m2(l1, m1, l2, m2)

                diff_3 = l1m1_sin_theta_cos_phi_l2m2_Lebedev(
                    l1, m1, l2, m2
                ) - l1m1_sin_theta_cos_phi_l2m2(l1, m1, l2, m2)

                diff_4 = l1m1_sin_theta_sin_phi_l2m2_Lebedev(
                    l1, m1, l2, m2
                ) - l1m1_sin_theta_sin_phi_l2m2(l1, m1, l2, m2)

                assert np.abs(diff) < 1e-12
                assert np.abs(diff_2) < 1e-12
                assert np.abs(diff_3) < 1e-12
                assert np.abs(diff_4) < 1e-12
