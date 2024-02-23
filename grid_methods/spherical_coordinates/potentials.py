import numpy as np
from scipy.special import erf


class Coulomb:
    def __init__(self, Z):
        self.Z = Z

    def __call__(self, r):
        return -self.Z / r


class Gaussian:
    def __init__(self, V0, alpha):
        self.V0 = V0
        self.alpha = alpha

    def __call__(self, r):
        return -self.V0 * np.exp(-self.alpha * r**2)


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


class SAE2:
    """
    Ref: 10.1088/2399-6528/ab9a68

    Params: See Table 1 in Ref.
    """

    def __init__(self, C0, Zc, c, a, b):
        self.C0 = C0
        self.Zc = Zc
        self.c = c
        self.a = a
        self.b = b
        self.n_a = len(a)

    def __call__(self, r):
        T1 = -self.C0 / r
        T2 = -self.Zc * np.exp(-self.c * r) / r
        T3 = np.zeros_like(r)

        for i in range(self.n_a):
            T3 -= self.a[i] * np.exp(-self.b[i] * r)

        return T1 + T2 + T3


class Erfgau:
    def __init__(self, Z, mu):
        self.Z = Z
        self.mu = mu

    def __call__(self, r):
        c = 0.923 + 1.568 * self.mu
        alpha = 0.2411 + 1.405 * self.mu
        long_range = erf(self.mu * self.Z * r) / (self.Z * r)
        return -self.Z**2 * (
            c * np.exp(-(alpha**2) * self.Z**2 * r**2) + long_range
        )
