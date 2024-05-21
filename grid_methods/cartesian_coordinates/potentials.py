import numpy as np


def Coulomb(x, Z, x_c=0.0, a=1.0):
    return -Z / np.sqrt((x - x_c) ** 2 + a**2)


class Molecule1D:
    def __init__(self, R=[0.0], Z=[1], alpha=1.0):
        """
        kwargs:
            R: list of nuclear positions
            Z: list of nuclear charges
            alpha: regularization parameter for the Coulomb potential
        """
        self.R_list = R
        self.Z_list = Z
        self.alpha = alpha

    def __call__(self, x):
        if isinstance(x, float):
            potential = 0
        else:
            potential = np.zeros(len(x))
        for R, Z in zip(self.R_list, self.Z_list):
            potential += Coulomb(x, Z=Z, x_c=R, a=self.alpha)
        return potential


def harmonic_oscillator(x, omega):
    return 0.5 * omega**2 * x**2
