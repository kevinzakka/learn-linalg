"""Single eigenpair finding algorithms.
"""

import numpy as np

from linalg import utils
from linalg.ludecomp import LU
from linalg.solver import solve


def power_iteration(A, max_iter=1000):
  """Finds the largest eigenpair of a symmetric matrix.

  Args:
    A: a square symmetric array of shape (N, N).

  Returns:
    e, v: eigenvalue and right eigenvector.
  """
  assert utils.is_symmetric(A), "[!] Matrix must be symmetric."
  v = np.random.randn(A.shape[0])
  for i in range(max_iter):
    v_new = A @ v
    v_new /= utils.l2_norm(v_new)
    if np.all(np.abs(v_new - v) < 1e-8):
      break
    v = v_new
  e = rayleigh_quotient(A, v)
  return e, v


def inverse_iteration(A, max_iter=1000):
  """Finds the smallest eigenpair of a symmetric matrix.

  Args:
    A: a square symmetric array of shape (N, N).

  Returns:
    e, v: eigenvalue and right eigenvector.
  """
  assert utils.is_symmetric(A), "[!] Matrix must be symmetric."
  v = np.random.randn(A.shape[0])
  PLU = LU(A, pivoting='partial').decompose()
  for i in range(max_iter):
    v_new = solve(PLU, v)
    v_new /= utils.l2_norm(v_new)
    if np.all(np.abs(v_new - v) < 1e-8):
      break
    v = v_new
  e = rayleigh_quotient(A, v)
  return e, v


def rayleigh_quotient_iteration(A, mu, max_iter=1000):
  """Finds an eigenpair closest to an initial eigenvalue guess.

  Args:
    A: a square symmetric array of shape (N, N).
    mu: an initial eigenvalue guess.

  Returns:
    e, v: eigenvalue and right eigenvector.
  """
  assert utils.is_symmetric(A), "[!] Matrix must be symmetric."
  v = np.random.randn(A.shape[0])
  for i in range(max_iter):
    v_new = solve(A - mu*np.eye(A.shape[0]), v)
    v_new /= utils.l2_norm(v_new)
    if np.all(np.abs(v_new - v) < 1e-8):
      break
    v = v_new
    mu = rayleigh_quotient(A, v)
  return mu, v


def rayleigh_quotient(A, x):
  """Computes the Rayleigh quotient.

  This is useful for determning an eigenvalue from
  an eigenvector, e.g. after using inverse iteration.
  """
  num = x.T @ A @ x
  denum = x.T @ x
  if np.isclose(denum, 1.):
    return num
  return num / denum
