"""A solver for systems of linear equations.
"""

import numpy as np

from linalg.ludecomp import LU


def solve(A_or_plu_or_pluq, b, pivoting='partial'):
  """Solves the linear system of equations Ax = b for a well-determined A.

  Concretely, uses PLU decomposition of A followed by forward
  and back substitution to solve for x.

  Args:
    A: a numpy array of shape (N, N) or a tuple containing the LU
      decomposition of the matrix A.
    b: a numpy array of shape (N,).
    pivoting: 'partial' or 'full' pivoting.

  Returns:
    x: a numpy array of shape (N, ).
  """
  if isinstance(A_or_plu_or_pluq, tuple):
    solver = LU(np.eye(A_or_plu_or_pluq[0].shape[0]), pivoting=pivoting)
    solver.set_P(A_or_plu_or_pluq[0])
    solver.set_L(A_or_plu_or_pluq[1])
    solver.set_U(A_or_plu_or_pluq[2])
    if len(A_or_plu_or_pluq) > 3:
      solver.set_Q(A_or_plu_or_pluq[3])
  else:
    M, N = A_or_plu_or_pluq.shape
    Z = len(b)

    error_msg = "[!] A must be square."
    assert (M == N), error_msg

    error_msg = "[!] b must be {}-D".format(M)
    assert (Z == N), error_msg

    solver = LU(A_or_plu_or_pluq, pivoting=pivoting)
    solver.decompose()

  # solve for x
  x = solver.solve(b)
  return x