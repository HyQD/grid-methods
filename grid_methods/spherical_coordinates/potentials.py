import numpy as np
from scipy.special import erf


class Coulomb:
    def __init__(self, Z):
        self.Z = Z

    def __call__(self, r):
        return -self.Z / r


class Gaussian_charge_distribution:
    def __init__(self, mu):
        self.mu = mu

    def __call__(self, r):
        return -erf(self.mu * r) / r


class SAE:
    def __init__(self, Z, A, B):
        """
        Ref: 10.1103/PhysRevA.74.053412

        ------------------
        Atom Z  A    B
        ------------------
        He   2  0.00 2.083
        Ar   18 5.40 3.682
        Xe   54 44.0 3.852
        ------------------
        ------------------
        """

        self.Z = Z
        self.A = A
        self.B = B

    def __call__(self, r):
        return (
            -1
            / r
            * (
                1
                + self.A * np.exp(-r)
                + (self.Z - 1 - self.A) * np.exp(-self.B * r)
            )
        )
