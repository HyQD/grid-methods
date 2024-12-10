import numpy as np

class Counter:
    # Used to count iterations until convergence in bicgstab
    def __init__(self):
        self.counter = 0

    def __call__(self, x):
        self.counter += 1


def quadrature(weights, f):
    return np.sum(weights * f)


def mask_function(r, r_max, r0, n=4):
    mask_r = np.zeros(len(r))

    ind1 = r < r0
    ind2 = r == r_max
    ind3 = np.invert(ind1 + ind2)

    mask_r[ind1] = 1
    mask_r[ind2] = 0
    mask_r[ind3] = np.cos(np.pi * (r[ind3] - r0) / (2 * (r_max - r0))) ** (
        1 / n
    )

    return mask_r
