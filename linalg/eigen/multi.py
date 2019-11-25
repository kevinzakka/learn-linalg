"""Algorithms for finding multiple eigenpairs.
"""

import numpy as np

from linalg import utils
from linalg.eigen.single import *


def projected_iteration(A, k, max_iter=1000):
  """Finds the k eigenpairs of a symmetric matrix.

  Args:
    A: a square symmetric array of shape (N, N).
    k: the number of eigenpairs to return.

  Returns:
    e, v: eigenvalue and eigenvector.
  """
  assert utils.is_symmetric(A), "[!] Matrix must be symmetric."
  assert k <= A.shape[0], "[!] k can be at most {}.".format(A.shape[0])
  assert k > 0, "[!] k must be greater than 0."
