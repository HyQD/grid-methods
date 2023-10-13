import numpy as np


class GaussChebyshevLobatto:
    """class ChebyshevLobato:
    Simple class for Chebyshef-Lobatto pseudospectral method and quadrature.
    """

    def __init__(self, a, b, N):
        self.set_grid(a, b, N)

    # def weight_function(self, x):
    #     xi =  2 * (x - self.a) / (self.b - self.a) - 1
    #     return (1.0 - xi**2)**(0.5)

    def quad_full(self, y):
        """Gaussian quadrature of y = y(x), a function evaluated on the grid."""
        return np.sum(self.qw * y)

    def quad(self, y):
        """Gaussian quadrature of y = y(x), a function evaluated on the grid."""
        return np.sum(self.qw[1:-1] * y)

    def set_grid(self, a, b, N):
        """Set the grid parameters and compute matrices etc.

        Sets the following important fields of the class:
        self.a # left endpoint of interval
        self.b # right endpoint of interval
        self.N # number of basis functions / nodes
        self.nodes # quadrature nodes
        self.qw # quadrature weights
        self.Dx # diff matrix on the grid
        self.T # coeff to grid
        self.M # grid to coeff
        """

        # Grid parameters
        assert N > 0
        self.a = a
        self.b = b
        self.N = N

        self.theta = np.pi * np.arange(N) / (N - 1)
        theta = self.theta
        self.nodes = np.cos(self.theta)
        self.T = np.zeros((N, N))
        self.Dx = np.zeros((N, N))
        self.w = np.ones((N,))
        self.w[0] = 0.5
        self.w[-1] = 0.5

        self.intT = np.zeros((N,))
        for i in range(N):
            if i == 1:
                self.intT[i] = 0
            else:
                self.intT[i] = ((-1) ** i + 1) / (1 - i**2)

        self.p = 1 / self.w
        p = self.p

        for i in range(N):
            self.T[:, i] = np.cos(i * theta)
            for j in range(N):
                if i == 0 and j == 0:
                    self.Dx[i, j] = (1 + 2 * (N - 1) ** 2) / 6
                elif i == N - 1 and j == N - 1:
                    self.Dx[i, j] = -(1 + 2 * (N - 1) ** 2) / 6
                elif i == j:
                    self.Dx[i, j] = -self.nodes[j] / (2 * (1 - self.nodes[j] ** 2))
                else:
                    self.Dx[i, j] = (
                        (-1) ** (i + j)
                        * p[i]
                        / (p[j] * (self.nodes[i] - self.nodes[j]))
                    )

        # scale domain and diff matrix
        self.nodes = (b - a) * (self.nodes + 1) / 2 + a
        self.Dx = 2 / (b - a) * self.Dx
        self.w /= ((N - 1) / 2) ** 0.5

        # Analysis matrix: For f = Tc, c = M f, where f is on the grid and c are coeffs.
        self.M = np.diag(self.w) @ self.T.T @ np.diag(self.w)  # analysis

        # Copute quadrature weights for computing definite integrals via self.quad.
        self.qw = self.M.T @ self.intT * (b - a) / 2


def quad_test():
    N = 15
    a = -4
    b = 2
    C = ChebyshevLobatto(a, b, N)
    x = C.x
    f = np.exp(x)
    integral_quad = C.quad(f)
    integral_exact = np.exp(b) - np.exp(a)
    print(
        f"Integral of exp(x) from {a} to {b} is {integral_quad}, the error is {integral_quad - integral_exact}."
    )


def quad_test2():
    N = 15
    a = 0
    b = 1
    C = ChebyshevLobatto(a, b, N)
    x = C.x

    for p in range(2 * N):
        f = (p + 1) * x**p
        integral_quad = C.quad(f)
        integral_exact = 1
        print(f"p = {p} integral error is {integral_quad - integral_exact}.")


if __name__ == "__main__":
    # Just a simple test of quadrature.
    # Compute definite integral of exp(x)

    quad_test()
    quad_test2()
