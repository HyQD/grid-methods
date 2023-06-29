import numpy as np
import matplotlib.pyplot as plt
from gauss_chebyshev_lobatto import GaussChebyshevLobatto
from scipy.special import erf

l = 0
r_max = 40
n_r = 100

C = GaussChebyshevLobatto(0, r_max, n_r)
r = C.nodes

r = r[
    1:-1
]  # Do not include the endpoints of the grid. We assume the psi(x0) = 0 and psi(xN) = 0
n_r = len(r)
potential = -1 / r

plt.figure()
plt.plot(r, -1 / r)
plt.plot(r, potential)

T = -0.5 * C.Dx @ C.Dx
T = T[1:-1, 1:-1] + np.diag(l * (l + 1) / (2 * r**2))
V = np.diag(potential)

H = T + V

# The Gauss-Chebyshev-Lobatto double-derivative matrix is not symmetric so we must use eig and not eigh
eps, psi = np.linalg.eig(H)

# Sort the eigenvalues and corresponding eigenvectors
idx = eps.argsort()
eps = eps[idx]
psi = psi[:, idx]
print(f"Ground state energy: {eps[0]}")
