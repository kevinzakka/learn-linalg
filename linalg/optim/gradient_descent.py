import numpy as np

from .base import IterativeSolver


class GradientDescent(IterativeSolver):
  """A gradient descent solver, a.k.a. steepest descent."""

  def _solve(self, A, b):
    n = A.shape[0]
    x = np.random.randn(n)  # Randomly initialize an estimate of x.
    for i in range(self.max_iters):
      d = b - A @ x  # Compute gradient with respect to x.
      alpha = (d.T @ d) / (d.T @ A @ d)  # Compute optimal step size.
      x_new = x + alpha * d  # Compute new estimate of x.
      if np.allclose(x_new, x, rtol=self.tol):
        print("Converged in {} iterations.".format(i))
        break
      x = x_new
    return x


# Alias.
SteepestDescent = GradientDescent
