"""A gradient descent algorithm for solving sparse, SPD systems Ax = b.
"""

import numpy as np


class GradientDescent:
  """A gradient descent solver, a.k.a. steepest descent.
  """
  def __init__(self, max_iters, tol=5*np.finfo(float).eps):
    """Constructor.

    Args:
      max_iters (int): The max number of iterations to run the
        solver for.
      tol (float): The tolerance for convergence.
    """
    self.max_iters = max_iters
    self.tol = tol

  def solve(self, A, b):
    n = A.shape[0]
    x = np.random.randn(n)  # initialize estimate of x
    for i in range(self.max_iters):
      d = b - A @ x  # compute gradient with respect to x
      alpha = (d.T @ d) / (d.T @ A @ d)  # compute optimal step size
      x_new = x + alpha * d  # compute new estimate of x
      if np.allclose(x_new, x, rtol=self.tol):
        print("Converged in {} iterations.".format(i))
        break
      x = x_new
    return x


SteepestDescent = GradientDescent