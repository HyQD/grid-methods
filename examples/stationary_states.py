import numpy as np
from grid_methods.spherical_coordinates.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Rational_map,
)

# Setup Legendre-Lobatto grid
N = 200
nr = N - 1
r_max = 40
alpha = 0.4

gll = GaussLegendreLobatto(N, Rational_map(r_max=r_max, alpha=alpha))
PN_x = gll.PN_x
weights = gll.weights

r = gll.r[1:-1]
D2 = gll.D2[1:-1, 1:-1]
ddr = -0.5 * D2

potential = -1 / r
V = np.diag(potential)

for l in range(0, 3):

    T = ddr + np.diag(l * (l + 1) / (2 * r**2))
    H_l = T + V

    eps_l, phi_nl = np.linalg.eigh(H_l)
    print(eps_l[0], eps_l[1], eps_l[2])
