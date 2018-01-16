from qr import QR
from lu import LU


def lstsq(A, b):
    """
    Solves the linear system of equations Ax = b by
    computing a vector x that minimizes the Euclidean
    norm ||b - Ax||^2.

    Concretely, uses the QR decomposition of A to deal
    with an over or well determined system.

    Params
    ------
    - A: a numpy array of shape (M, N).
    - b: a numpy array of shape (M,).

    Returns:
    - x: a numpy array of shape (N, ).
    """
    M, N = A.shape

    # if well-determined, use PLU
    if (M == N):
        solver = LU(A, pivoting='partial')
    else:
        solver = QR(A)

    # solve for x
    x = solver.solve(b)

    return x
