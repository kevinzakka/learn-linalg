import numpy as np

from functools import reduce


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
