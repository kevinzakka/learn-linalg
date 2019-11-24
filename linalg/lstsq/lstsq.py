"""A least-squares solver for systems of linear equations.
"""

from qrdecomp.qr import QR
from ludecomp.lu import LU


def lstsq(A, b):
  """Finds the least-squares solution to a linear system Ax = b for an over-determined A.

  Solves the linear system of equations Ax = b by computing a vector
  x that minimizes the Euclidean norm ||b - Ax||^2 using QR decomposition.

  Args:
    A: a numpy array of shape (M, N).
    b: a numpy array of shape (M,).

  Returns:
    x: a numpy array of shape (N, ).
  """
  M, N = A.shape

  # if well-determined, use PLU
  if (M == N):
    solver = LU(A, pivoting='partial')
  else:
    solver = QR(A)

  # solve for x
  x = solver.solve(b)

  return x