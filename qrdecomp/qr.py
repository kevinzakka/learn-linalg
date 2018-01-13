import numpy as np




def QR(object):
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
        self.A = np.array(A)

    def decompose(self):
        """
        Starting initially with Gram-Schmidt.

        Q = AE_1E_2...E_k
        R = (E_1E_2...E_k).inv
        """
        self.Q = self.A

