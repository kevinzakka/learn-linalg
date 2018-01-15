import numpy as np

from sum import KahanSum
from functools import reduce


def reflection(x, u, apply=False):
    """
    A reflection is a linear transformation
    which reflects a vector x with respect to
    a hyperplane through the origin represented
    by its normal vector v of unit length.

    Concretely, the reflection can be considered
    a linear transformation represented by a matrix
    P, i.e. `x' = Px`.

    Params
    ------
    - x: A numpy array of shape (N, 1). The column
      vector to reflect.
    - u: A numpy array of shape (N, 1). The unnormalized
      vector normal to the hyperplane.
    - apply: bool indicating whether to apply the reflecton
      directly and obtain x' or return the matrix P if set
      to `False`.

    Returns
    -------
    - refl: a numpy array of shape (N, 1) if apply is `True`.
    - P: a numpy array of shape (N, N) if apply is `False`.
    """
    # grab dimension of column vector
    N = x.shape[0]

    # then normalize u
    v = u / l2_norm(u)

    if apply:
        # compute projection of x onto v
        proj = projection(x, v, norm=True)

        # and finally reflect
        refl = x - 2*proj

        return refl
    else:
        P = np.eye(N) - 2*np.dot(v, v.T)
        return P


def projection(b, a, norm=False):
    """
    The projection of b onto a is the orthogonal
    projection of b onto a straight line parallel to a.

    The projection is parallel to a, i.e. it is the product
    of a constant called the scalar projection with a unit
    vector in the direction of a:

    `proj(b, a) = (c)a = (a.Tb/a.Ta)a`

    Params
    ------
    - b: a numpy array of shape (N, 1).
    - a: a numpy array of shape (N, 1).
    - norm: bool indicating whether a is normalized
      or not.

    Returns
    -------
    - proj: a numpy array of shape (N, 1).
    """
    if norm:
        proj = np.dot(np.dot(a, a.T), b)
    else:
        c = np.dot(a.T, b) / np.dot(a.T, a)
        proj = c * a

    return proj


def l2_norm(x):
    """
    L2 or "euclidean" norm.
    """
    return np.sqrt(np.dot(x.T, x))


def norm(x, p):
    """
    Returns the p norm of a vector.
    """
    v = np.array(x).flatten()

    error_msg = "x must be 1D"
    assert v.ndim == 1, error_msg
    error_msg = "p must be >= 1"
    assert p >= 1, error_msg

    N = v.shape[0]

    summer = KahanSum()
    for i in range(N):
        summer.add(np.power(np.abs(v[i]), p))

    return np.power(summer.cur_sum(), 1./p)


def herm(A):
    """
    Returns the conjugate transpose of A.
    Equivalent to the H operator `A.H`.
    """
    return A.T.conj()


def is_hermitian(A):
    """
    Returns True if A is hermitian.
    """
    return np.allclose(A, A.T.conj())


def is_symmetric(A):
    """
    Returns True if A is symmetric.
    """
    return np.allclose(A, A.T)


def upper_diag(A, diag=False):
    """
    Grabs the super-diagonal elements of a
    square matrix A.
    """
    m = len(A)
    U = np.zeros_like(A)

    for i in range(m):
        l_b = i + 1
        if diag:
            l_b = i
        for j in range(l_b, m):
            U[i, j] = A[i, j]

    return U


def lower_diag(A, diag=False):
    """
    Grabs the sub-diagonal elements of a
    square matrix A.
    """
    m = len(A)
    L = np.zeros_like(A)

    for i in range(m):
        u_b = i
        if diag:
            u_b = i + 1
        for j in range(0, u_b):
            L[i, j] = A[i, j]

    return L


def unit_diag(A):
    """
    Fills the diagonal elements of a
    square matrix A with 1's.
    """
    m = len(A)

    for i in range(m):
        A[i, i] = 1

    return A


def multi_dot(l):
    """
    Performs consecutive dot products of the
    arrays in the list l from left to right.

    For example, given l = [A, B, C], returns
    `np.dot(np.dot(A, B), C)`.

    """
    return reduce(np.dot, l)


def basis_vec(k, n):
    """
    Creates the k'th standard basis vector in R^n.
    """

    error_msg = "[!] k cannot exceed {}.".format(n)
    assert (k < n), error_msg

    b = np.zeros([n, 1])
    b[k] = 1
    return b


def basis_arr(ks, n):
    """
    Creates an array of k'th standard basis vectors in R^n
    according to each k in ks.
    """

    error_msg = "[!] ks cannot exceed {}.".format(n)
    assert (np.max(ks) < n), error_msg

    b = np.zeros([n, n])
    for i, k in enumerate(ks):
        b[i, k] = 1
    return b
