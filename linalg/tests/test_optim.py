import unittest
import numpy as np

from linalg.cholesky import Cholesky
from linalg.optim import GradientDescent, ConjugateGradient
from linalg.utils import random_spd


class OptimTest(unittest.TestCase):
  """Tests various iterative linear solvers.
  """
  def test_gradient_descent(self):
    A = random_spd(50)
    b = np.random.randn(50)

    expected = Cholesky(A).solve(b)
    actual = GradientDescent(1000000).solve(A, b)

    self.assertTrue(np.allclose(expected, actual))

  def test_conjugate_gradient(self):
    pass


if __name__ == "__main__":
  unittest.main()
