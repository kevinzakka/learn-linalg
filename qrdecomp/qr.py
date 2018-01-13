import numpy as np

from utils import proj, l2_norm


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

    def decompose(self):
        """
        Starting initially with Gram-Schmidt.
        """
        self.Q = np.array(self.A)
        self.R = np.zeros_like(self.A)

        M, N = self.Q.shape

        # for each column
        for i in range(N):
            # for each column on the left of current
            for j in range(i):
                # calculate projection of ith col on jth col
                p = proj(self.Q[:, i], self.Q[:, j])
                self.Q[:, i] -= p
            # normalize ith colum
            self.Q[:, i] /= l2_norm(self.Q[:, i])

        # compute R
        self.R = np.dot(self.Q.T, self.A)

        return self.Q, self.R


