import abc

import numpy as np

from linalg.eigen.utils import is_spd


class IterativeSolver(abc.ABC):
  """An iterative solver for SPD systems Ax=b.

  Subclasses must implement `_solve`.
  """

  def __init__(self, max_iters, tol=5*np.finfo(float).eps):
    """Constructor.

    Args:
      max_iters (int): The max number of iterations to run the solver for.
      tol (float): The tolerance for convergence.
    """
    self.max_iters = max_iters
    self.tol = tol

  @abc.abstractmethod
  def _solve(self, A, b):
    pass

  def solve(self, A, b):
    """Solve the SPD system Ax=b."""
    assert is_spd(A), "[!] A must be symmetric positive definite."
    return self._solve(A, b)
