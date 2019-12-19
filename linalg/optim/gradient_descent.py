"""A gradient descent algorithm for solving Ax = b.
"""

import numpy as np


class GradientDescent:
  """A gradient descent solver.
  """
  def __init__(self, max_iters, tol=1e-9):
    """Constructor.

    Args:
      max_iters (int): The max number of iterations to run the
        solver for.
      tol (float): The tolerance for convergence. Specifically, if
        the squared norm of the difference between consecutive
        estimates of x decreases below this tolerance, then the
        iterations are stopped.
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
      if ((x_new - x).T @ (x_new - x)) <= self.tol:
        print("Converged in {} iterations.".format(i))
        break
      x = x_new
    return x
