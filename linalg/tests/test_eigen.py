import unittest
import numpy as np
import numpy.linalg as LA

from linalg.eigen import single


class EigenTest(unittest.TestCase):
  """Tests eigenvalue finding algorithms.
  """
  def same_eigvec(self, x, y):
    if np.sign(x[0]) == np.sign(y[0]):
      diff = x - y
    else:
      diff = x + y
    return np.allclose(diff, np.zeros(3))

  def test_power_iteration(self):
    M = np.random.randn(3, 3)
    M = M.T @ M

    eigvals, eigvecs = LA.eig(M)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]
    expected_eigvec = eigvecs[:, 0]
    expected_eigval = eigvals[0]

    actual_eigval, actual_eigvec = single.power_iteration(M, 10000)

    self.assertTrue(self.same_eigvec(actual_eigvec, expected_eigvec))
    self.assertTrue(np.isclose(actual_eigval, expected_eigval))

  def test_inverse_iteration(self):
    M = np.random.randn(3, 3)
    M = M.T @ M

    eigvals, eigvecs = LA.eig(M)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]
    expected_eigvec = eigvecs[:, -1]
    expected_eigval = eigvals[-1]

    actual_eigval, actual_eigvec = single.inverse_iteration(M, 10000)

    self.assertTrue(self.same_eigvec(actual_eigvec, expected_eigvec))
    self.assertTrue(np.isclose(actual_eigval, expected_eigval))

  def test_rayleigh_largest(self):
    M = np.random.randn(3, 3)
    M = M.T @ M

    eigvals, eigvecs = LA.eig(M)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]
    expected_eigvec = eigvecs[:, 0]
    expected_eigval = eigvals[0]

    initial_eigval = single.power_iteration(M, 50)[0]
    actual_eigval, actual_eigvec = single.rayleigh_quotient_iteration(M, initial_eigval)

    self.assertTrue(self.same_eigvec(actual_eigvec, expected_eigvec))
    self.assertTrue(np.isclose(actual_eigval, expected_eigval))

  def test_rayleigh_smallest(self):
    M = np.random.randn(3, 3)
    M = M.T @ M

    eigvals, eigvecs = LA.eig(M)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]
    expected_eigvec = eigvecs[:, -1]
    expected_eigval = eigvals[-1]

    initial_eigval = single.inverse_iteration(M, 50)[0]
    actual_eigval, actual_eigvec = single.rayleigh_quotient_iteration(M, initial_eigval)

    self.assertTrue(self.same_eigvec(actual_eigvec, expected_eigvec))
    self.assertTrue(np.isclose(actual_eigval, expected_eigval))


if __name__ == '__main__':
  unittest.main()