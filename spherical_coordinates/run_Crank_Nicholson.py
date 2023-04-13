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
#potential = erf(5 * r) / r
#potential = -10 * np.exp(-0.1 * r ** 2)

plt.figure()
plt.plot(r, -potential)

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
plt.plot(time_points, expec_z.real)

plt.figure()
plt.subplot(211)
plt.plot(r, np.abs(psi_tfinal_r) ** 2)
plt.subplot(212)
plt.plot(r, psi_tfinal[0].real)
plt.plot(r, psi_tfinal[0].imag)

plt.show()

# h5f = h5py.File(f'dat/{potential_type}_E0={E0}_omega={omega}_nc={nc}_dt={dt}_rmax={r_max}_dr={dr}_lmax={l_max}.h5', 'w')
# # #h5f.create_dataset('psi_t_r', data=psi_t)
# h5f.create_dataset('r', data=r)
# h5f.create_dataset('time_points', data=time_points)
# h5f.create_dataset('expec_z', data=expec_z)
# h5f.create_dataset('laser_t', data=laser(time_points))
# h5f.create_dataset('potential', data=potential)
# h5f.create_dataset('psi_t_final', data=psi_t)
# h5f.close()
