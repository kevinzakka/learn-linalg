"""Algorithms for finding multiple eigenpairs.
"""

import numpy as np

from linalg import utils
from linalg.eigen.single import *


def projected_iteration(A, k, max_iter=1000):
  """Sequentially find the k eigenpairs of a symmetric matrix.

  Concretely, combines power iteration and deflation to find
  eigenpairs in order of decreasing magnitude.

  Args:
    A: a square symmetric array of shape (N, N).
    k: the number of eigenpairs to return.

  Returns:
    e, v: eigenvalue and eigenvector.
  """
  assert utils.is_symmetric(A), "[!] Matrix must be symmetric."
  assert k <= A.shape[0], "[!] k can be at most {}.".format(A.shape[0])
  assert k > 0, "[!] k must be greater than 0."
  pass


def qr_algorithm(A):
  pass
