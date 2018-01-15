import numpy as np

from utils import projection, l2_norm


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

        for i in range(N):
            

    def gram_schmidt(self):
        """
        Compute QR using Gram-Schmidt orthogonalization.

        Suffers from numerical instabilities when 2 vectors
        are nearly orthogonal so not really used in practice.
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
