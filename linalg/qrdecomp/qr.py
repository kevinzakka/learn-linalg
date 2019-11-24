"""QR decomposition.
"""

import numpy as np

from linalg import utils
from linalg.kahan.sum import KahanSum


class QR(object):
    """Computes the QR decomposition of an `mxn` matrix A.

    The matrix A is factorized into a product of an
    orthogonal matrix Q and an upper-triangular matrix R.

    QR decomposition is widely used for solving the linear
    least squares problem `Ax=b` when A is overdetermined
    (m > n).

    Args:
        A: a numpy array of shape (M, N).
        b: a column vector of shape (M,). Can also be multiple column
            vectors in the case where one wishes to solve many right-hand
            sides associated with the same A. In that case, b is of shape
            (M, K).
        reduce: bool indicating whether to return the reduced QR factorization.

    Returns:
        If Gram-Schmidt:
            Q: (M, N)
            R: (N, N)
        If Householder, `reduce=False`:
            Q: (M, M)
            R: (M, N)
        If Householder, `reduce=True`:
            Q: (M, N)
            R: (N, N)
    """
    def __init__(self, A, reduce=False):
        self.backup = np.array(A, dtype=np.float64)
        self.reduce = reduce

    def decompose(self):
        Q, R = self.householder()
        return Q, R

    def householder(self, ret=True):
        """Computes QR using Householder triangularization.
        """
        self.A = np.array(self.backup)
        M, N = self.A.shape
        self.Q = np.eye(M)

        if M == N:
            iters = N - 1
        else:
            iters = N

        for i in range(iters):
            # select column
            c = self.A[i:M, i:i+1]

            # grab sign of first element in c
            s = np.sign(c[0])

            # compute u
            u = c + s*utils.l2_norm(c)*utils.basis_vec(0, M-i)

            # reflect the submatrix with respect to u
            self.A[i:M, i:N] = utils.reflection(self.A[i:M, i:N], u, apply=True)

            # update Q (need to make more efficient)
            Q = utils.reflection(self.A[i:M, i:N], u, apply=False)
            Q = np.pad(Q, ((i, 0), (i, 0)), mode='constant')
            for j in range(i):
                Q[j, j] = 1.
            self.Q = np.dot(Q, self.Q)

        self.Q = self.Q.T
        self.R = self.A

        if self.reduce:
            self.Q = self.Q[:, :N]
            self.R = self.R[:N, :]

        if ret:
            return self.Q, self.R

    def gram_schmidt(self):
        """Computes QR using Gram-Schmidt orthogonalization.

        Suffers from numerical instabilities when 2 vectors
        are nearly orthogonal which may cause loss of
        orthogonality between the columns of Q.

        We compute a unique QR decomposition, hence we force
        the diagonal elements of R to be positive.
        """
        self.A = np.array(self.backup)
        M, N = self.A.shape
        self.R = np.array(self.A)

        # for each column
        for i in range(N):
            # for each column on the left of current
            for j in range(i):
                # calculate utils.projection of ith col on jth col
                p = utils.projection(self.A[:, i], self.A[:, j])
                self.A[:, i] -= p
            # normalize ith column
            self.A[:, i] /= utils.l2_norm(self.A[:, i])

        # Q is now A, R = (Q.T)(A)
        self.Q = self.A
        self.R = np.dot(self.Q.T, self.R)

        return self.Q, self.R

    def solve(self, b):
        """Solves the lineary system Ax = b.

        Specifically, performs the QR factorization of A,
        then solves Ax = b using backward substitution.
        """
        self.b = b

        self.householder(ret=False)
        self._backward()

        return self.x

    def _backward(self):
        """Solve the upper triangular system Rx = y
        for x by back substitution, where `y = Q.Tb`.
        """
        M, N = self.R.shape
        self.y = np.dot(self.Q.T, self.b)

        if self.y.ndim == 1:
            self.y = np.reshape(self.y, [-1, 1])

        if self.b.ndim == 2:
            num_iters = self.b.shape[1]
        else:
            num_iters = 1

        self.x = np.zeros([N, num_iters])

        for k in range(num_iters):
            for i in range(N-1, -1, -1):
                acc = KahanSum()
                for j in range(N-1, i, -1):
                    acc.add(self.R[i, j]*self.x[j, k])
                self.x[i, k] = (self.y[i, k] - acc.cur_sum()) / (self.R[i, i])

        if self.b.ndim == 1:
            self.x = self.x.squeeze()