import numpy as np

from .base import IterativeSolver


class ConjugateGradient(IterativeSolver):
  """A conjugate gradient (CG) solver.

  Unlike gradient descent, CG is guaranteed to converge in at most `n` steps.
  """

  def __init__(self, max_iters, tol=1e-6):
    self.max_iters = max_iters
    self.tol = tol

  def _solve(self, A, b):
    n = A.shape[0]
    x = np.random.randn(n)  # Randomly initialize an estimate of x.
    r = b - A @ x
    v = np.array(r, copy=True)
    beta = 0
    for i in range(self.max_iters):
      v = r + beta*v  # Search direction.
      alpha = r@r / (v.T @ A @ v)  # Line search.
      x = x + alpha*v  # Update estimate.
      r_old = r  # Save the old residual.
      r = r - alpha*A@v  # Update the residual.
      if (r@r) < self.tol*(r_old@r_old):
        print("Converged in {} iterations.".format(i))
        break
      beta = (r@r) / (r_old@r_old)  # Direction step.
    return x
