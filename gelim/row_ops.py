import numpy as np

from utils import basis_vec, basis_arr


def permute(N, idx):
    """
    Permutes the rows of a square matrix A of shape (N, N)
    according to a list of indices stored in idx.

    idx can be of 2 forms:

    - a flat list that must contain all the new row orders.
      e.g. if A is (3, 3), then idx = [2, 1, 0]
    - a nested list of pairs (x, y) that signifies "switch row
      x with row y". Any unmentioned rows stay the same.
      e.g. if A is (6, 6) and idx = [(0, 3), (1, 2)], then this
      translates to idx = [3, 2, 1, 0, 4, 5].

    Args
    ----
    - N: number of rows/columns of square matrix A.
    - idx: a list of integers or integer pairs where
      max(idx) < (N - 1).

    Returns
    -------
    - P: permutation array of size (N, N).
    """

    # check if list is nested
    nested = any(isinstance(i, (tuple, list)) for i in idx)

    # convert to standard form
    if nested:

        # ensure that idx does not exceed size of A
        flat_idx = [i for sublist in idx for i in sublist]
        error_msg = "[!] Indices cannot exceed {}.".format(N)
        assert (np.max(flat_idx) < N), error_msg

        before = [i[0] for i in idx]
        after = [i[1] for i in idx]

        idx = list(np.arange(N))
        for i in range(len(before)):
            idx[before[i]], idx[after[i]] = idx[after[i]], idx[before[i]]

    # ensure that idx does not exceed size of A
    error_msg = "[!] Indices cannot exceed {}.".format(N)
    assert (np.max(idx) < N), error_msg

    # construct permutation matrix
    P = basis_arr(idx, N)

    return P


def scale(N, scalars):
    """
    Scales the rows of a square matrix A of shape (N, N)
    according to a list of non-zero scalrs stored in the
    list scalars.

    scalars can be of 2 forms:

    - a flat list that must contain all the row scalars.
      e.g. if A is (3, 3), then idx = [1, 1, 0.5].
    - a nested list of pairs (r, a) that signifies
      "scale row r by a value fo a". Any unmentioned rows
      are assumed to be scaled by a value of 1.
      e.g. if A is (6, 6) and scalars = [(0, 3), (1, 2)], then this
      translates to scalars = [3, 2, 1, 1, 1, 1].

    Args
    ----
    - N: number of rows/columns of square matrix A.
    - scalars: a list of non-zero integers to scale by.

    Returns
    -------
    - S: scaling array of size (N, N).
    """

    # check if list is nested
    nested = any(isinstance(i, (tuple, list)) for i in scalars)

    # convert to standard form
    if nested:

        row_nums = [i[0] for i in scalars]
        row_scalars = [i[1] for i in scalars]

        # ensure that row numbers do not exceed length of A
        error_msg = "[!] Indices cannot exceed {}.".format(N)
        assert (np.max(row_nums) < N), error_msg

        scalars = [1] * N
        for row, scalar in zip(row_nums, row_scalars):
            scalars[row] = scalar

    # ensure that scalars are non zero
    error_msg = "[!] Scalars cannot be 0."
    assert (0 not in scalars), error_msg

    # construct scaling matrix
    S = np.diag(scalars)

    return S


def eliminate(N, k, c, l):
    """
    Scales row k of a square matrix A of shape (N, N)
    by c and adds the result to row l.

    Args
    ----
    - N: number of rows/columns of square matrix A.
    - c: a non-zero integer to scale row k by.
    - k: an intger where A[k] is the row to scale by c.
    - l: an integer where A[l] += c * A[k]

    Returns
    -------
    - E: elimination array of size (N, N).
    """

    # ensure scalar is non zero
    error_msg = "[!] Scalar cannot be 0."
    assert (c != 0), error_msg

    # ensure l and k do not exceed rows of A
    error_msg = "[!] Indices cannot exceed {}.".format(N)
    assert ((l and k) < N), error_msg

    # construct elimination matrix
    row_k = basis_vec(k, N)
    row_l = basis_vec(l, N)
    E = np.eye(N) + (c * np.dot(row_l, row_k.T))

    return E
