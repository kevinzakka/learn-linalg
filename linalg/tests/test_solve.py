import unittest
import numpy as np
import numpy.linalg as LA

from scipy.linalg import lu_factor, lu_solve
from linalg.solver import solve
from linalg.ludecomp import LU


class SolveTest(unittest.TestCase):
  """Tests solve for a well determined system.
  """
  def test_well_determined_partial(self):
    A = np.random.randn(50, 50)
    b = np.random.randn(50)

    actual = solve(A, b, 'partial')
    expected = LA.solve(A, b)

    self.assertTrue(np.allclose(actual, expected))

  def test_well_determined_full(self):
    A = np.random.randn(50, 50)
    b = np.random.randn(50)

    actual = solve(A, b, 'full')
    expected = LA.solve(A, b)

    self.assertTrue(np.allclose(actual, expected))

  def test_already_factored_partial(self):
    A = np.random.randn(2, 2)
    b = np.random.randn(2)

    P, L, U = LU(A, pivoting='partial').decompose()
    actual = solve((P, L, U), b, 'partial')

    lu, piv = lu_factor(A)
    expected = lu_solve((lu, piv), b)

    self.assertTrue(np.allclose(actual, expected))


if __name__ == '__main__':
  unittest.main()