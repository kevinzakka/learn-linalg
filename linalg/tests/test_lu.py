import unittest
import numpy as np
import scipy.linalg as LA

from linalg.ludecomp import LU
from linalg.utils import multi_dot


class LUDecompositionTest(unittest.TestCase):
  """Tests LU decomposition with various pivoting strategies.
  """
  def test_no_pivoting(self):
    T = np.random.randn(3, 3)

    L_a, U_a = LU(T).decompose()
    actual = np.dot(L_a, U_a)

    self.assertTrue(np.allclose(actual, T))

  def test_partial_pivoting(self):
    T = np.random.randn(50, 50)

    actual = LU(T, pivoting='partial').decompose()
    expected = LA.lu(T)

    self.assertTrue(all(np.allclose(a, e) for a, e in zip(actual, expected)))

  def test_full_pivoting(self):
    T = np.random.randn(50, 50)

    actual = list(LU(T, pivoting='full').decompose())

    self.assertTrue(np.allclose(multi_dot(actual), T))

  def test_solve_single_no_pivoting(self):
    T = np.random.randn(50, 50)
    b = np.random.randn(50)

    lu_solver = LU(T)
    lu_solver.decompose()
    actual = lu_solver.solve(b)
    expected = np.linalg.solve(T, b)

    self.assertTrue(np.allclose(actual, expected))

  def test_solve_multi_no_pivoting(self):
    T = np.random.randn(50, 50)
    b = np.random.randn(50, 5)

    lu_solver = LU(T)
    lu_solver.decompose()
    actual = lu_solver.solve(b)
    expected = np.linalg.solve(T, b)

    self.assertTrue(np.allclose(actual, expected))

  def test_solve_multi_partial_pivoting(self):
    T = np.random.randn(50, 50)
    b = np.random.randn(50, 5)

    lu_solver = LU(T, pivoting='partial')
    lu_solver.decompose()
    actual = lu_solver.solve(b)
    expected = np.linalg.solve(T, b)

    self.assertTrue(np.allclose(actual, expected))

  def test_solve_multi_full_pivoting(self):
    T = np.random.randn(50, 50)
    b = np.random.randn(50, 5)

    lu_solver = LU(T, pivoting='full')
    lu_solver.decompose()
    actual = lu_solver.solve(b)
    expected = np.linalg.solve(T, b)

    self.assertTrue(np.allclose(actual, expected))


if __name__ == '__main__':
  unittest.main()
