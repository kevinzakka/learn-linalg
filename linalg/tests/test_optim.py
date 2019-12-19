import unittest
import numpy as np

from linalg.lstsq import lstsq
from linalg.optim import GradientDescent, ConjugateGradient


class OptimTest(unittest.TestCase):
  """Tests various iterative linear solvers.
  """
  def test_gradient_descent(self):
    A = np.random.randn(30, 30)
    A = A.T @ A  # make it symmetric
    b = np.random.randn(30)

    expected = lstsq(A, b)
    actual = GradientDescent(1000000, 1e-15).solve(A, b)

    self.assertTrue(np.allclose(expected, actual, rtol=1e-2))

  def test_conjugate_gradient(self):
    pass


if __name__ == "__main__":
  unittest.main()
