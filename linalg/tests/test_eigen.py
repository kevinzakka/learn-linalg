import unittest
import numpy as np
import numpy.linalg as LA

from linalg.eigen import single
from linalg.utils import random_symmetric


class EigenTest(unittest.TestCase):
  """Tests eigenvalue finding algorithms.
  """
  def same_eigvec(self, x, y):
    if np.sign(x[0]) == np.sign(y[0]):
      diff = x - y
    else:
      diff = x + y
    return np.allclose(diff, np.zeros_like(diff))

  def np_eig(self, M, largest=True):
    eigvals, eigvecs = LA.eig(M)
    # sort by largest absolute eigenvalue
    idx = np.abs(eigvals).argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]
    if largest:
      return eigvals[0], eigvecs[:, 0]
    return eigvals[-1], eigvecs[:, -1]

  def test_power_iteration(self):
    M = random_symmetric(3)

    expected_eigval, expected_eigvec = self.np_eig(M)

    actual_eigval, actual_eigvec = single.power_iteration(M, 1000)

    self.assertTrue(self.same_eigvec(actual_eigvec, expected_eigvec))
    self.assertTrue(np.isclose(actual_eigval, expected_eigval))

  def test_inverse_iteration(self):
    M = random_symmetric(3)

    expected_eigval, expected_eigvec = self.np_eig(M, largest=False)

    actual_eigval, actual_eigvec = single.inverse_iteration(M, 10000)

    self.assertTrue(self.same_eigvec(actual_eigvec, expected_eigvec))
    self.assertTrue(np.isclose(actual_eigval, expected_eigval))

  def test_rayleigh_largest(self):
    M = random_symmetric(3)

    expected_eigval, expected_eigvec = self.np_eig(M)

    initial_eigval = single.power_iteration(M, 50)[0]
    actual_eigval, actual_eigvec = single.rayleigh_quotient_iteration(M, initial_eigval)

    self.assertTrue(self.same_eigvec(actual_eigvec, expected_eigvec))
    self.assertTrue(np.isclose(actual_eigval, expected_eigval))

  def test_rayleigh_smallest(self):
    M = random_symmetric(3)

    expected_eigval, expected_eigvec = self.np_eig(M, largest=False)

    initial_eigval = single.inverse_iteration(M, 50)[0]
    actual_eigval, actual_eigvec = single.rayleigh_quotient_iteration(M, initial_eigval)

    self.assertTrue(self.same_eigvec(actual_eigvec, expected_eigvec))
    self.assertTrue(np.isclose(actual_eigval, expected_eigval))


if __name__ == '__main__':
  unittest.main()
