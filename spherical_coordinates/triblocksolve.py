import numpy as np
import time


def block_tridiag_solve(lower, diagonal, upper, b):
    """
    Solve Ax = b where A is a block tridiagonal matrix

    A = |D1, U1,  0,  0,  0, ..., 0|
        |L1, D2, U2,  0,  0, ..., 0|
        | 0, L2, D2, U3,  0, ..., 0|
        | 0,  0, L3, D2, U4, ..., 0|
    """

    # Get number of blocks
    nblk = diagonal.shape[0]
    N = diagonal.shape[1]

    # Store solution as matrix
    x = np.zeros((nblk, N), dtype=diagonal.dtype)
    c = x.copy()

    Q = diagonal.copy()
    G = diagonal.copy()

    Q[0] = diagonal[0].copy()
    G[0] = np.linalg.solve(Q[0], upper[0])

    for k in range(1, nblk - 1):
        Q[k] = diagonal[k] - np.dot(lower[k - 1], G[k - 1])
        G[k] = np.linalg.solve(Q[k], upper[k])

    Q[nblk - 1] = diagonal[nblk - 1] - np.dot(lower[nblk - 2], G[nblk - 2])

    c[0] = np.linalg.solve(Q[0], b[0])

    for k in range(1, nblk):
        c[k] = np.linalg.solve(Q[k], b[k] - np.dot(lower[k - 1], c[k - 1]))

    x[nblk - 1] = c[nblk - 1].copy()

    for k in range(nblk - 2, -1, -1):
        x[k] = c[k] - np.dot(G[k], x[k + 1])

    return x
