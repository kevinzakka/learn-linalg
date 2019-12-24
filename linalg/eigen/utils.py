import numpy as np

from linalg.eigen.multi import eig
from linalg.utils import is_symmetric


def is_spd(A):
  """Returns True if A is symmetric positive definite.
  """
  assert A.ndim == 2, "Expecting 2D matrix."
  if not is_symmetric(A):
    return False
  eigvals, _ = eig(A, sort=False)
  return np.all(eigvals >= 0)