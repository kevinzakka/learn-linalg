import numpy as np

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

    Returns
    -------
    - L: a lower triangular numpy array of shape (N, N).
    """

    def __init__(self, A):
        self.R = upper_diag(np.array(A), diag=True)

        # check that A is symmetric
        error_msg = 'A must be symmetric!'
        assert is_symmetric(A), error_msg

    def decompose(self):
        """
        In this first attempt, I'll be restricting myself to
        real matrices, hence symmetric ones.

        I'll also implement an inefficient way of computing the
        Cholesky factorization wherein I apply an elimination
        matrix on the left than on the right.

        [Done]

        A more efficient version now:

        We know the matrix is symmetric and that the row containing the
        pivot will be eliminated just like the column underneath the
        pivot. So we can focus on just the upper diagonal part of A
        and ignore the lower part.

        Also, instead of creating an elimination matrix, we just create
        another loop to eliminate element-wise.

        So like in LU decomposition, we edit A in place with the scaling
        factors. Since our first column is already 0 due to just taking the
        upper diagonal, then the values in the first row are already the values
        we would have scaled by to eliminate the column. We just need to post scale
        them by the inverse of the square root of the pivot.

        Thus, at every row, we eliminate the rows beneath just like in LU and just
        scale the current row by the square root since those are the values of the
        lower triangular matrix that would have been applied leftwise to eliminate
        the column.

        [Done]
        """

        N = len(self.R)

        for i in range(N):
            self.pivot = self.R[i, i]

            # eliminate subsequent rows
            for j in range(i+1, N):
                for k in range(j, N):
                    self.R[j, k] -= self.R[i, k] * (self.R[i, j] / self.pivot)

            # scale the current row
            for k in range(i, N):
                self.R[i, k] /= np.sqrt(self.pivot)

        return self.R
