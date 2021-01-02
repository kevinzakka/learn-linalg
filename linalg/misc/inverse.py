import numpy as np

from linalg.ludecomp import LU
from linalg.kahan import KahanSum


def inverse(A):
  """Computes the inverse of a square matrix A.

  Concretely, solves the linear system Ax = I
  where x is a square matrix rather than a vector.

  The system is solved using LU decomposition with
  partial pivoting.

  Args:
    A: a numpy array of shape (N, N).

  Returns:
    A numpy array of shape (N, N).
  """
  N = A.shape[0]

  P, L, U = LU(A, pivoting='partial').decompose()

  # transpose P since LU returns A = PLU
  P = P.T

  # solve Ly = P for y
  y = np.zeros_like(L)
  for i in range(N):
    for j in range(N):
      summer = KahanSum()
      for k in range(i):
        summer.add(L[i, k]*y[k, j])
      sum = summer.result()
      y[i, j] = (P[i, j] - sum) / (L[i, i])

  # solve Ux = y for x
  x = np.zeros_like(U)
  for i in range(N-1, -1, -1):
    for j in range(N):
      summer = KahanSum()
      for k in range(N-1, i, -1):
        summer.add(U[i, k]*x[k, j])
      sum = summer.result()
      x[i, j] = (y[i, j] - sum) / (U[i, i])

  return x
