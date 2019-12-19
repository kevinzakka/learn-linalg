import unittest
import numpy as np
import numpy.linalg as LA

from linalg.cholesky import Cholesky


class CholeskyDecompositionTest(unittest.TestCase):
  """Tests Cholesky decomposition.
  """
  def test_decompose_no_crout(self):
    T = np.array([
      [4, 12, -16],
      [12, 37, -43],
      [-16, -43, 98]
    ])

    actual = Cholesky(T, crout=False).decompose()
    expected = LA.cholesky(T)

    self.assertTrue(np.allclose(actual, expected))

  def test_decompose_crout(self):
    T = np.array([
      [4, 12, -16],
      [12, 37, -43],
      [-16, -43, 98]
    ])

    actual = Cholesky(T, crout=True).decompose()
    expected = LA.cholesky(T)

    self.assertTrue(np.allclose(actual, expected))

  def test_solve_single(self):
    T = np.array([
      [4, 12, -16],
      [12, 37, -43],
      [-16, -43, 98]
    ])
    b = np.array([1, 2, 3])

    actual = Cholesky(T).solve(b)
    expected = np.linalg.solve(T, b)

    self.assertTrue(np.allclose(actual, expected))

  def test_solve_multi(self):
    T = np.array([
      [4, 12, -16],
      [12, 37, -43],
      [-16, -43, 98]
    ])
    b = np.array([[1, 2, 3], [2, 1, 2]]).T

    actual = Cholesky(T).solve(b)
    expected = np.linalg.solve(T, b)

    self.assertTrue(np.allclose(actual, expected))


if __name__ == '__main__':
  unittest.main()
