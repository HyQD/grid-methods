import numpy as np
from utils import mask_function
from lasers import sine_square_laser
from matplotlib import pyplot as plt
import tqdm
import sys
from CrankNicholson import CrankNicholson_sine_DVR
from scipy.special import erf

E0 = float(sys.argv[1])
omega = float(sys.argv[2])
nc = int(sys.argv[3])
dt = float(sys.argv[4])

r_max = float(sys.argv[5])
dr = float(sys.argv[6])
l_max = int(sys.argv[7])

n_r = int(r_max / dr)
r = np.arange(1, n_r + 1) * dr


t_cycle = 2 * np.pi / omega
tfinal = nc * t_cycle
print(f"Tfinal: {tfinal}")

laser = sine_square_laser(E0=E0, omega=omega, td=tfinal)

num_steps = int(np.ceil(tfinal) / dt)
time_points = np.arange(0, num_steps) * dt

sampling_rate = int(np.ceil(1.0 / dt))


r_cut = 32
absorber = np.zeros(n_r)
for k in range(n_r):
    absorber[k] = mask_function(r[k], r_max, r_max - r_cut)


potential = 1 / r
# potential = erf(5 * r) / r
# potential = -10 * np.exp(-0.1 * r ** 2)

expec_z, psi_tfinal = CrankNicholson_sine_DVR(
    r,
    potential,
    l_max,
    time_points,
    laser,
    absorber=[True, absorber],
    dump_psi_t=[False, 100, "dat"],
)

psi_tfinal_r = np.zeros(n_r, dtype=psi_tfinal.dtype)
for l in range(l_max + 1):
    psi_tfinal_r += psi_tfinal[l]


plt.figure()
plt.plot(r, -potential, label=r"$V(r)$")
plt.legend()
plt.xlabel("r[a.u.]")

plt.figure()
plt.plot(time_points, expec_z.real, label=r"$\langle z(t) \rangle$")
plt.legend()
plt.xlabel("Time[a.u.]")


plt.figure()
plt.plot(r, np.abs(psi_tfinal_r) ** 2, label=r"$|\Psi(r, t_{final})|^2$")
plt.legend()
plt.xlabel("r[a.u.]")

plt.show()
