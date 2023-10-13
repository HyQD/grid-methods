import numpy as np
from matplotlib import pyplot as plt
from scipy.special import sph_harm
import quadpy
import math
import time
from utils import *
from lasers import sine_square_laser, linear_laser
import tqdm
from scipy.integrate import complex_ode
from gauss_integrator import GaussIntegrator
from rk4_integrator import Rk4Integrator
from opt_einsum import contract

scheme = quadpy.u3.schemes["lebedev_029"]()
theta, phi = scheme.theta_phi
weights = scheme.weights


def kron_delta(x1, x2):
    if x1 == x2:
        return 1
    else:
        return 0


def Yl1m1_costheta_Yl2m2(l1, m1, l2, m2):
    Y_m1_l1 = sph_harm(m1, l1, phi, theta)
    Y_m2_l2 = sph_harm(m2, l2, phi, theta)

    integrand = Y_m1_l1.conj() * np.cos(theta) * Y_m2_l2
    integral = np.sum(4 * np.pi * weights * integrand)

    if abs(integral.imag) > 1e-10:
        print(f"Warning, the integral is complex valued.")

    return integral


def Yl1m1_sin2theta_Yl2m2(l1, m1, l2, m2):
    Y_m1_l1 = sph_harm(m1, l1, phi, theta)
    Y_m2_l2 = sph_harm(m2, l2, phi, theta)

    integrand = Y_m1_l1.conj() * np.sin(theta) ** 2 * Y_m2_l2
    integral = np.sum(4 * np.pi * weights * integrand)

    return integral


def get_T_element(k1, k2, dr):
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


r_max = 10
dr = 0.25
n_r = int(r_max / dr)
r = np.arange(1, n_r + 1) * dr
B = 0.5
m = 0

l_max = 5
dim = n_r * (l_max + 1 - abs(m))

T = np.zeros((n_r, n_r))
for k1 in range(n_r):
    for k2 in range(n_r):
        T[k1, k2] = get_T_element(k1 + 1, k2 + 1, dr)

G = np.zeros((l_max + 1, l_max + 1))
V_angular = np.zeros((l_max + 1, l_max + 1))

for l1 in range(0, l_max + 1):
    for l2 in range(0, l_max + 1):
        G[l1, l2] = Yl1m1_sin2theta_Yl2m2(l1, m, l2, m).real
        V_angular[l1, l2] = Yl1m1_costheta_Yl2m2(l1, m, l2, m).real

H = np.zeros((dim, dim))
V_int = np.zeros((dim, dim))
row = 0
for l1 in range(abs(m), l_max + 1):
    for k1 in range(n_r):
        col = 0
        for l2 in range(abs(m), l_max + 1):
            for k2 in range(n_r):
                H[row, col] = T[k1, k2] * kron_delta(l1, l2)
                H[row, col] += (
                    l2
                    * (l2 + 1)
                    / (2 * r[k2] ** 2)
                    * kron_delta(k1, k2)
                    * kron_delta(l1, l2)
                )
                H[row, col] += -1 / r[k2] * kron_delta(k1, k2) * kron_delta(l1, l2)
                # H[row, col] += (
                #     B * m / 2 * kron_delta(k1, k2) * kron_delta(l1, l2)
                # )
                H[row, col] += B**2 / 8 * r[k2] ** 2 * G[l1, l2] * kron_delta(k1, k2)

                V_int[row, col] += r[k2] * V_angular[l1, l2] * kron_delta(k1, k2)

                col += 1
        row += 1

eps, C_all = np.linalg.eigh(H)
print(eps[0])

C0 = np.complex128(C_all[:, 0])

E0 = 0.001
omega = 0.057

t_cycle = 2 * np.pi / omega
td = t_cycle


laser = sine_square_laser(E0=E0, omega=omega, td=td)

tfinal = td
dt = 0.01

num_steps = int(tfinal / dt) + 1
print(f"number of time steps: {num_steps}")

time_points = np.linspace(0, tfinal, num_steps)
dipole_moment = np.zeros(num_steps, dtype=np.complex128)


def rhs(t, C):
    Ht = H + laser(t) * V_int
    HC = np.dot(Ht, C)
    return -1j * HC


propagator = complex_ode(rhs).set_integrator("GaussIntegrator", s=3, eps=1e-10)
propagator.set_initial_value(C0)

for i in tqdm.tqdm(range(num_steps - 1)):
    propagator.integrate(propagator.t + dt)
    C_t = propagator.y.reshape(l_max + 1, n_r)
    dipole_moment[i + 1] = contract("lk, mk, k, lm->", C_t.conj(), C_t, r, V_angular)

np.save("time-points", time_points)
np.save(f"dvr-dip-mom-B={B}", dipole_moment)

plt.figure()
plt.plot(time_points, dipole_moment.real)
plt.show()
