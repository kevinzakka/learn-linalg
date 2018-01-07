import numpy as np

from lu import LU
from utils import diagonal, multi_dot


def det(X, log=False):
    """
    Computes the determinant of a square matrix A.

    Concretely, first factorizes A into PLU and then computes
    the product of the determinant of P and U.

    In case where the determinant is a very small or very big
    number, the implementation may underflow or overflow. To combat
    this, we compute the log of the determinant and return the sign
    in which case:

    `det = sign * np.exp(logdet)`

    Args
    ----
    - A: a numpy array of shape (N, N).
    - log: set to True to return the log of the determinant
      and the sign.

    Returns
    -------
    If log = False, returns:

    - det: a scalar, the determinant of A.

    Else, returns a tuple:

    - sign: 1 or -1.
    - logdet: a float representing the log of the determinant.
    """
    A = np.array(X)

    # LU decomposition
    t, U = LU(A, pivoting='partial').decompose(det=True)

    # compute determinant of P
    if t % 2 == 0:
        sign = 1.
    else:
        sign = -1.

    # compute determinant of U and then A
    diag = diagonal(U)
    if log:
        logdet = 0.
        for d in diag:
            logdet += np.log(d)
    else:
        det_U = multi_dot(diag)
        det_A = sign * det_U

    if log:
        return sign, logdet
    return det_A
