import numpy as np
from numpy.polynomial import legendre
from scipy.special import eval_legendre as Legendre


class GaussLegendreLobatto:
    def __init__(self, x0, xN, N_tot):
        self.x0 = x0
        self.xN = xN
        self.N_tot = N_tot
        self.set_grid(self.x0, self.xN, self.N_tot)

    def quad_bp(self, y):
        return np.sum(self.qw * y)

    def quad(self, y):
        """Gaussian quadrature of y = y(x), a function evaluated on the grid."""
        return np.sum(self.qw[1:-1] * y)

    def set_grid(self, x0, xN, N_tot):
        """
        Compute nodes, weights and differentiation matrix for Gauss-Legendre-Lobatto
        quadrature on the interval [a, b].

        Parameters
        ----------
        a :
            left endpoint of the interval
        b :
            right endpoint of the interval
        N_tot :
            The total number of nodes/grid points including the endpoints.

        Notes
        -----


        """

        # Grid parameters
        assert N_tot > 0

        ##########################################################
        N = N_tot - 1

        # Get inner nodes
        c = np.zeros((N + 1,))
        c[-1] = 1
        dc = legendre.legder(c)

        self.nodes = np.zeros(N + 1)
        self.nodes[0] = -1
        self.nodes[-1] = 1
        self.nodes[1:-1] = legendre.legroots(dc)

        # Check for correct number of roots, and uniqueness
        assert len(self.nodes[1:-1]) == N - 1
        assert np.allclose(np.unique(self.nodes[1:-1]), self.nodes[1:-1])

        # Compute PN(x_i) and the weights
        self.PN_x = legendre.legval(self.nodes, c)
        self.weights = 2 / (N * (N + 1) * self.PN_x**2)
        ############################################################

        self.Dx = np.zeros((N_tot, N_tot))
        for i in range(N_tot):
            for j in range(N_tot):
                if i == 0 and j == 0:
                    self.Dx[i, j] = 1 / 4 * (N_tot - 1) * N_tot
                elif i == N_tot - 1 and j == N_tot - 1:
                    self.Dx[i, j] = -1 / 4 * (N_tot - 1) * N_tot
                elif (i == j) and (0 < j < N_tot - 1):
                    self.Dx[i, j] = 0
                else:
                    self.Dx[i, j] = self.PN_x[i] / (
                        self.PN_x[j] * (self.nodes[i] - self.nodes[j])
                    )

        # scale domain and diff matrix
        self.nodes = (xN - x0) * (self.nodes + 1) / 2 + x0
        self.Dx = 2 / (xN - x0) * self.Dx
        self.qw = self.weights * (xN - x0) / 2
