import numpy as np
from scipy.integrate import simps
from matplotlib import pyplot as plt
from scipy.integrate import complex_ode
from gauss_integrator import GaussIntegrator
from lasers import sine_square_laser, linear_laser
from utils import compute_numerical_states

delta = lambda x, y: x == y

# This is part of the second integral in the dipole matrix element expression
def c_l(l):
    if l < 0:
        return 0
    else:
        return (l + 1) / np.sqrt((2*l + 1) * (2*l + 3))

def second_integral(l_i, l_j):
    return c_l(l_i)*delta(l_i+1,l_j)+c_l(l_i-1)*delta(l_i-1,l_j)


# Define some stuff
r_max = 100
dr = 0.25
n_grid = int(r_max / dr)
r = np.arange(1, n_grid + 1) * dr

n_max = 20
l_max = 4

eigenenergies, eigenstates = compute_numerical_states(l_max, n_max, r)

# Allocate space for matrix elements
H0 = np.zeros([n_max*l_max]*2)
H_int = np.zeros([n_max*l_max]*2)
I2 = np.zeros([l_max]*2)

for i in range(l_max):
    for j in range(l_max):
        I2[i, j] = second_integral(i, j)


# Create mapping i -> (l, n) (n is not quantum number)
mapping = {}
i = 0
for l in range(l_max):
    for n in range(n_max):
        mapping[i] = (l, n)
        print(f"{i} -> {(n,l)}, E_{n,l} = {eigenenergies[l][n]}")
        i += 1

# Compute H_int
for i in range(n_max*l_max):
    l_i, n_i = mapping[i]
    u_i = eigenstates[l_i][:, n_i]
    for j in range(n_max*l_max):
        l_j, n_j = mapping[j]
        u_j = eigenstates[l_j][:, n_j]
        H0[i,j] = eigenenergies[l_i][n_i]*delta(i,j)
        H_int[i, j] = simps(u_i*r*u_j, r) * second_integral(l_i, l_j) 


E0 = 0.03  # 0.02387
omega = 0.057  # 0.04286

t_cycle = 2 * np.pi / omega
td = 3 * t_cycle

laser = sine_square_laser(
    E0=E0, omega=omega, td=td
)  # linear_laser(E0=E0, omega=omega, n_ramp=5)


def rhs(t, C):
    Ht = H0 + laser(t) * H_int
    return -1j * np.dot(Ht, C)


n_basis = H0.shape[0]

C = np.complex128(np.eye(n_basis))
C0 = C[:, 0].copy()

r = complex_ode(rhs).set_integrator("GaussIntegrator", s=3, eps=1e-10)
r.set_initial_value(C0)

tfinal = td
dt = 0.1

num_steps = int(tfinal / dt) + 1
time_points = np.linspace(0, tfinal, num_steps)

dipole_moment = np.zeros(num_steps, dtype=np.complex128)
dipole_moment[0] = np.vdot(C0, H_int @ C0)
overlaps = np.zeros((num_steps, n_basis), dtype=np.complex128)
norm = np.zeros(num_steps, dtype=np.complex128)
norm[0] = np.vdot(C0, C0)

for j in range(n_basis):
    overlaps[0, j] = np.vdot(C[:, j], C0)


for i in range(num_steps - 1):

    r.integrate(r.t + dt)
    dipole_moment[i + 1] = np.vdot(r.y, H_int @ r.y)
    for j in range(n_basis):
        overlaps[i + 1, j] = r.y[j]
    norm[i + 1] = np.vdot(r.y, r.y)
    if i % 100 == 0:
        print(i)



np.save("time-points-num-orbs", time_points)
np.save("num-orbs-dip-mom", dipole_moment)

plt.figure()
plt.plot(time_points, -dipole_moment.real)


plt.figure()
plt.plot(time_points, np.abs(overlaps[:,0]) ** 2)
plt.legend()

plt.figure()
plt.plot(time_points, 1 - np.abs(norm) ** 2)

plt.show()