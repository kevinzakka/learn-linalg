import numpy as np

from sum import KahanSum
from utils import is_symmetric, upper_diag


class Cholesky(object):
    """
    Computes the Cholesky factorization of a symmetric
    positive definite (SPD) matrix A.

    Instead of seeking arbitrary lower and upper triangular
    factors L and U, the Cholesky factorization exploits the
    symmetry properties of A to construct a lower triangular
    matrix L whose transpose can serve as the upper triangular
    part. When used on SPD matrices, Cholesky will run twice as
    fast as LU.

    Cholesky factorization analogously works on the complex
    equivalent of symmetric matrices, Hermitian matrices, in
    which case it returns L and its conjugate transpose L.H.

    Note that the decomposition will fail if the matrix is not
    symmetric or Hermitian.

    Args
    ----
    - A: a numpy array of shape (N, N).
    - crout: Whether to use the Crout algorithm. From my
      experience, setting it to False yields a faster
      factorization.

    Returns
    -------
    - L: a lower triangular numpy array of shape (N, N).
    """

    def __init__(self, A, crout=True):
        self.crout = crout
        if self.crout:
            self.R = np.array(A)
        else:
            self.R = upper_diag(np.array(A), diag=True)

        # check that A is symmetric
        error_msg = 'A must be symmetric!'
        assert is_symmetric(A), error_msg

        # add checking for positive definite
        # this will be done by checking eigenvalues positive
        # will do this once relevant section has been studied

    def decompose(self, ret=True):

        N = len(self.R)

        self.L = np.zeros_like(self.R)

        if self.crout:
            for i in range(N):
                for j in range(i+1):
                    summer = KahanSum()
                    for k in range(j):
                        summer.add(self.L[i, k] * self.L[j, k])
                    sum = summer.cur_sum()

                    if (i == j):
                        self.L[j, j] = np.sqrt(self.R[j, j] - sum)
                    else:
                        self.L[i, j] = (self.R[i, j] - sum) / (self.L[j, j])
            if ret:
                return self.L
        else:
            for i in range(N):
                self.pivot = self.R[i, i]

                # eliminate subsequent rows
                for j in range(i+1, N):
                    for k in range(j, N):
                        self.R[j, k] -= self.R[i, k] * \
                                        (self.R[i, j] / self.pivot)

                # scale the current row
                for k in range(i, N):
                    self.R[i, k] /= np.sqrt(self.pivot)

            if ret:
                return self.R.T

    def solve(self, b):
        """
        Perform the Cholesky factorization on the
        matrix A and then solve the linear system
        Ax = b using forward and backward substitution.
        """
        self.b = b

        self.decompose(ret=False)

        if not self.crout:
            self.L = self.R.T

        self._forward()
        self._backward()

        return self.x

    def _forward(self):
        """
        Solves the lower triangular system Ly = b
        for y by forward substitution.
        """
        if self.b.ndim > 1:
            num_iters = self.b.shape[1]
            N = self.b.shape[0]
        else:
            num_iters = 1
            N = self.b.shape[0]

        self.y = np.zeros([N, num_iters])

        for k in range(num_iters):
            for i in range(N):
                acc = KahanSum()
                for j in range(i):
                    acc.add(self.L[i, j]*self.y[j, k])
                if self.b.ndim > 1:
                    self.y[i, k] = \
                        (self.b[i, k] - acc.cur_sum()) / (self.L[i, i])
                else:
                    self.y[i, k] = \
                        (self.b[i] - acc.cur_sum()) / (self.L[i, i])

    def _backward(self):
        """
        Solve the upper triangular system L^Tx = y
        for x by back substitution.
        """
        if self.b.ndim > 1:
            num_iters = self.b.shape[1]
            N = self.b.shape[0]
        else:
            num_iters = 1
            N = self.b.shape[0]

        self.x = np.zeros([N, num_iters])

        for k in range(num_iters):
            for i in range(N-1, -1, -1):
                acc = KahanSum()
                for j in range(N-1, i, -1):
                    acc.add(self.L.T[i, j]*self.x[j, k])
                self.x[i, k] = \
                    (self.y[i, k] - acc.cur_sum()) / (self.L.T[i, i])

        if self.b.ndim == 1:
            self.x = self.x.squeeze()
