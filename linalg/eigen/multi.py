"""Algorithms for finding multiple eigenpairs.
"""

import numpy as np

from linalg import utils
from linalg.eigen.single import rayleigh_quotient


def projected_iteration(A, k, max_iter=1000, sort=True):
  """Sequentially find the k eigenpairs of a symmetric matrix.

  Concretely, combines power iteration and deflation to find
  eigenpairs in order of decreasing magnitude.

  Args:
    A: a square symmetric array of shape (N, N).
    k: the number of eigenpairs to return.
    sort: Whether to sort by decreasing eigenvalue magnitude.

  Returns:
    e, v: eigenvalues and eigenvectors. The eigenvectors are
      stacked column-wise.
  """
  assert utils.is_symmetric(A), "[!] Matrix must be symmetric."
  assert k > 0 and k <= A.shape[0], "[!] k must be between 1 and {}.".format(A.shape[0])

  eigvecs = np.zeros((A.shape[0], k))
  eigvals = np.zeros(A.shape[0])
  for i in range(k):
    v = np.random.randn(A.shape[0])
    for poop in range(max_iter):
      # project out computed eigenvectors
      proj_sum = np.zeros_like(v)
      for j in range(i):
        proj_sum += utils.projection(v, eigvecs[:, j])
      v -= proj_sum

      v = A @ v
      v /= utils.l2_norm(v)
    e = rayleigh_quotient(A, v)

    # store eigenpair
    eigvecs[:, i] = np.array(v)
    eigvals[i] = e

  # sort by largest absolute eigenvalue
  if sort:
    idx = np.abs(eigvals).argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]

  return eigvals, eigvecs


def qr_algorithm(A):
  raise NotImplementedError


def eig(A, sort=True):
  """Compute the eigenvalues and right eigenvectors of a symmetric matrix.
  """
  eigvals, eigvecs = projected_iteration(A, len(A), sort=sort)
  return eigvals, eigvecs