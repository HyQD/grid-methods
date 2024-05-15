import numpy as np


def Coulomb(x, Z, a=1.0):
    return -Z / np.sqrt(x**2 + a**2)


def harmonic_oscillator(x, omega):
    return 0.5 * omega**2 * x**2
