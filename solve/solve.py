from lu import LU


def solve(A, b, pivoting='partial'):
    """
    Solves the linear system of equations Ax = b
    where A is well-determined.

    Concretely, uses PLU decompisition of A
    followed by forward and back substitution
    to solve for x.

    Params
    ------
    - A: a numpy array of shape (N, N).
    - b: a numpy array of shape (N,).
    - pivoting: 'partial' or 'full' pivoting.

    Returns:
    - x: a numpy array of shape (N, ).
    """
    M, N = A.shape
    Z = len(b)

    error_msg = "[!] A must be square."
    assert (M == N), error_msg

    error_msg = "[!] b must be {}D".format(M)
    assert (Z == N), error_msg

    solver = LU(A, pivoting=pivoting)

    # solve for x
    x = solver.solve(b)

    return x
