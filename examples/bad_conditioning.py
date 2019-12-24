"""This script illustrates how Ax=b can be unstable
when the columns of A are nearly linearly dependent
when solved by factorizations that make use of the
the gram matrix `A.T @ A`, e.g. Cholesky factorization.

This instability is related to the condition number of
the matrix A. In fact, the condition number of `A.T @ A`
is the square of the condition number of A which explains
why strategies that use the normal equation can exhibit
considerable numerical error.
"""

import numpy as np
import numpy.linalg as LA

from linalg.cholesky import Cholesky
from linalg import inverse


if __name__ == "__main__":
  A = np.array([
    [0.01, 0.010001, .5],
    [0.02, 0.020001, 0.],
    [0.03, 0.030001, 0.5],
  ])
  T = A.T @ A
  b = np.array([.1, 2., .3])

  actual = Cholesky(T).solve(b)
  expected = LA.solve(T, b)
  diff = np.linalg.norm(actual - expected)
  cond = np.linalg.norm(A) * np.linalg.norm(inverse(A))
  print("Condition number {:.2f} - solution difference: {}".format(cond, diff))

  A = np.random.randn(3, 3)
  T = A.T @ A
  actual = Cholesky(T).solve(b)
  expected = LA.solve(T, b)
  diff = np.linalg.norm(actual - expected)
  cond = np.linalg.norm(A) * np.linalg.norm(inverse(A))
  print("Condition number {:.2f} - solution difference: {}".format(cond, diff))