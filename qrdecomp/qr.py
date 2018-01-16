import numpy as np

from utils import l2_norm, basis_vec
from utils import reflection, projection


class QR(object):
    """
    Computes the QR decomposition of an mxn matrix A. This
    is useful for solving linear systems of equations of the
    form Ax = b for non-square matrices, i.e. least squares.

    Params
    ------
    - A: a numpy array of shape (M, N).

    Returns
    -------
    - Q: a numpy array of shape (M, N).
    - R: a numpy array of shape (N, N).
    """

    def __init__(self, A):
        self.A = np.array(A, dtype=np.float64)
        self.Q = np.array(self.A)
        self.R = np.zeros_like(self.A)

    def householder(self):
        """
        Compute QR using Householder triangularization.
        """
        M, N = self.A.shape

        self.Q = np.eye(N)

        for i in range(N-1):
            # select column
            c = self.A[i:M, i:i+1]

            # grab sign of first element in c
            s = np.sign(c[0])

            # compute u
            u = c + s*l2_norm(c)*basis_vec(0, N-i)

            # reflect the submatrix with respect to u
            self.A[i:M, i:N] = reflection(
                self.A[i:M, i:N], u, apply=True
            )

            Q = reflection(self.A[i:M, i:N], u, apply=False)
            Q = np.pad(Q, ((i, 0), (i, 0)), mode='constant')
            for j in range(i):
                Q[j, j] = 1.
            self.Q = np.dot(Q, self.Q)

        return self.Q.T, self.A

    def gram_schmidt(self):
        """
        Compute QR using Gram-Schmidt orthogonalization.

        Suffers from numerical instabilities when 2 vectors
        are nearly orthogonal which may cause loss of
        orthogonality between the columns of Q.

        We compute a unique QR decomposition, hence we force
        the diagonal elements of R to be positive.
        """
        M, N = self.Q.shape

        # for each column
        for i in range(N):
            # for each column on the left of current
            for j in range(i):
                # calculate projection of ith col on jth col
                p = projection(self.Q[:, i], self.Q[:, j])
                self.Q[:, i] -= p
            # normalize ith column
            self.Q[:, i] /= l2_norm(self.Q[:, i])

        # compute R
        self.R = np.dot(self.Q.T, self.A)

        return self.Q, self.R
