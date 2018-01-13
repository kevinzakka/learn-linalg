import numpy as np

from cholesky import Cholesky
from sum import KahanSum


def pinv(A):
    """
    Computes the Moore-Penrose inverse of an mxn
    matrix A.

    For the best accuracy, this is usually computed
    using SVD but I have not implemented it yet. I
    will instead solve the least squares solution
    to Ax = b using the normal equation and Cholesky
    decomposition.

    To be changed to SVD at a later time.

    Params
    ------
    - A: a numpy array of shape (M, N).

    Returns
    -------
    - a numpy array of shape (N, M).
    """
    N = A.shape[0]

    P, L, U = LU(A, pivoting='partial').decompose()

    # transpose P since LU returns A = PLU
    P = P.T

    # solve Ly = P for y
    y = np.zeros_like(L)
    for i in range(N):
        for j in range(N):
            summer = KahanSum()
            for k in range(i):
                summer.add(L[i, k]*y[k, j])
            sum = summer.cur_sum()
            y[i, j] = (P[i, j] - sum) / (L[i, i])

    # solve Ux = y for x
    x = np.zeros_like(U)
    for i in range(N-1, -1, -1):
        for j in range(N):
            summer = KahanSum()
            for k in range(N-1, i, -1):
                summer.add(U[i, k]*x[k, j])
            sum = summer.cur_sum()
            x[i, j] = (y[i, j] - sum) / (U[i, i])

    return x

