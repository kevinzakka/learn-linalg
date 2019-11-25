"""Single eigenvalue finding algorithms.
"""

import numpy as np

from linalg import utils
from linalg.ludecomp import LU
from linalg.solver import solve


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
  v = np.random.randn(A.shape[0])
  PLU = LU(A, pivoting='partial').decompose()
  for i in range(max_iter):
    v = solve(PLU, v)
    v /= utils.l2_norm(v)
  return v


def rayleigh_quotient(A, x):
  """Computes the Rayleigh quotient.

  This is useful for determning an eigenvalue from
  an eigenvector, e.g. after using inverse iteration.
  """
  num = A @ x @ x
  denum = x @ x
  return num / denum
