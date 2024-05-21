import numpy as np


def Coulomb(x, Z, x_c=0.0, a=1.0):
    """
    Coulomb potential in 1D.

    Args:
        x (np.ndarray): The grid points.
        Z (float): The nuclear charge.
        x_c (float): The nuclear position.
        a (float): The regularization parameter.
    """
    return -Z / np.sqrt((x - x_c) ** 2 + a**2)


class Molecule1D:
    def __init__(self, R=[0.0], Z=[1], alpha=1.0):
        """
        Molecular potential in 1D.

        Args:
            R (list): The nuclear positions.
            Z (list): The nuclear charges.
            alpha (float): The regularization parameter.
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
