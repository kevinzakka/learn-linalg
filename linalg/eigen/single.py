"""Single eigenvalue finding algorithms.
"""

import numpy as np

from linalg import utils, inverse


def power_iteration(A, max_iter=1000):
  """Finds the largest eigenvector of a symmetric matrix.

  Args:
    A: a square symmetric array of shape (N, N).

  Returns:
    eigvec: the largest eigenvector of the matrix.
  """
  assert utils.is_symmetric(A), "[!] Matrix must be symmetric."
  v = np.random.randn(A.shape[0])
  for i in range(max_iter):
    v = A @ v
    v /= utils.l2_norm(v)
  return v


def inverse_iteration(A, max_iter=1000):
  """Finds the smallest eigenvector of a symmetric matrix.

  Args:
    A: a square symmetric array of shape (N, N).

  Returns:
    eigvec: the smallest eigenvector of the matrix.
  """
  assert utils.is_symmetric(A), "[!] Matrix must be symmetric."
  A_inv = inverse(A)
  v = np.random.randn(A.shape[0])
  for i in range(max_iter):
    v = A_inv @ v
    v /= utils.l2_norm(v)
  return v
