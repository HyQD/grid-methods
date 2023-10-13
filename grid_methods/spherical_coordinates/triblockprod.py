import numpy as np


def block_tridiag_product(lower, diagonal, upper, x):
    """
    Compute the matrix-vector product v = Ax where A is a block-tridiagonal matrix on the form

    A = |D1, U1,  0,  0,  0, ..., 0|
    |L1, D2, U2,  0,  0, ..., 0|
    | 0, L2, D2, U3,  0, ..., 0|
    | 0,  0, L3, D2, U4, ..., 0|


    """

    L = diagonal.shape[0]
    N = diagonal.shape[1]

    # x = x.reshape((L, N))
    v = np.zeros(x.shape, dtype=diagonal.dtype)

    v[0] = np.dot(diagonal[0], x[0]) + np.dot(upper[0], x[1])

    for l in range(1, L - 1):
        v[l] = (
            np.dot(lower[l - 1], x[l - 1])
            + np.dot(diagonal[l], x[l])
            + np.dot(upper[l], x[l + 1])
        )

    v[L - 1] = np.dot(lower[L - 2], x[L - 2]) + np.dot(diagonal[L - 1], x[L - 1])

    return v
