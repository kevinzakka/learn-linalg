import unittest
import numpy as np
import numpy.linalg as LA

from linalg.svd import SVD
from linalg import utils


class SVDTest(unittest.TestCase):
  """Tests Singular Value Decomposition.
  """
  def test_factorization_equals_initial_symmetric(self):
    A = utils.random_symmetric(3)

    U, S, V = SVD(A).decompose()

    actual = U @ S @ V.T
    expected = A

    self.assertTrue(np.allclose(actual, expected))

  def test_factorization_equals_initial_non_symmetric(self):
    A = np.random.randn(7, 5)

    U, S, V = SVD(A).decompose()

    actual = U @ S @ V.T
    expected = A

    self.assertTrue(np.allclose(actual, expected))


if __name__ == '__main__':
  unittest.main()