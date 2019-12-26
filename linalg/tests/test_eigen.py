import unittest
import numpy as np
import numpy.linalg as LA

from scipy.linalg import hessenberg as hessenberg_scipy
from linalg.eigen import single, multi
from linalg.utils import random_symmetric, is_symmetric


class EigenTest(unittest.TestCase):
  """Tests eigenvalue finding algorithms.
  """
  def absallclose(self, x, y, rtol=1e-5):
    return np.allclose(np.abs(x), np.abs(y), rtol=rtol)

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

    self.assertTrue(self.absallclose(actual_eigvec, expected_eigvec))
    self.assertTrue(self.absallclose(actual_eigval, expected_eigval))

  def test_inverse_iteration(self):
    M = random_symmetric(3)

    expected_eigval, expected_eigvec = self.np_eig(M, largest=False)

    actual_eigval, actual_eigvec = single.inverse_iteration(M, 10000)

    self.assertTrue(self.absallclose(actual_eigvec, expected_eigvec))
    self.assertTrue(self.absallclose(actual_eigval, expected_eigval))

  def test_rayleigh_largest(self):
    M = random_symmetric(3)

    expected_eigval, expected_eigvec = self.np_eig(M)

    initial_eigval = single.power_iteration(M, 50)[0]
    actual_eigval, actual_eigvec = single.rayleigh_quotient_iteration(M, initial_eigval)

    self.assertTrue(self.absallclose(actual_eigvec, expected_eigvec))
    self.assertTrue(self.absallclose(actual_eigval, expected_eigval))

  def test_rayleigh_smallest(self):
    M = random_symmetric(3)

    expected_eigval, expected_eigvec = self.np_eig(M, largest=False)

    initial_eigval = single.inverse_iteration(M, 50)[0]
    actual_eigval, actual_eigvec = single.rayleigh_quotient_iteration(M, initial_eigval)

    self.assertTrue(self.absallclose(actual_eigvec, expected_eigvec))
    self.assertTrue(self.absallclose(actual_eigval, expected_eigval))

  def test_projected_iteration(self):
    M = random_symmetric(3)

    eigvals, eigvecs = LA.eig(M)
    idx = np.abs(eigvals).argsort()[::-1]
    expected_eigvecs = eigvecs[:, idx]
    expected_eigvals = eigvals[idx]

    actual_eigvals, actual_eigvecs = multi.projected_iteration(M, 3)

    self.assertTrue(self.absallclose(actual_eigvecs, expected_eigvecs))
    self.assertTrue(self.absallclose(actual_eigvals, expected_eigvals))

  def test_hessenberg_h(self):
    M = np.random.rand(10, 10)

    actual_hess = multi.hessenberg(M, calc_q=False)
    expected_hess = hessenberg_scipy(M)

    self.assertTrue(np.allclose(actual_hess, expected_hess))

  def test_hessenberg_q(self):
    M = np.random.rand(10, 10)

    _, actual_Q = multi.hessenberg(M, calc_q=True)
    _, expected_Q = hessenberg_scipy(M, calc_q=True)

    self.assertTrue(np.allclose(actual_Q, expected_Q))

  def test_hessenberg_symmetric_stays_symmetric(self):
    M = random_symmetric(4)

    hess = multi.hessenberg(M)
    self.assertTrue(is_symmetric(hess))

  def test_qr_algorithm(self):
    M = random_symmetric(4)

    eigvals, eigvecs = LA.eig(M)
    idx = np.abs(eigvals).argsort()[::-1]
    expected_eigvecs = eigvecs[:, idx]
    expected_eigvals = eigvals[idx]

    actual_eigvals, actual_eigvecs = multi.qr_algorithm(M, sort=True)

    self.assertTrue(self.absallclose(actual_eigvecs, expected_eigvecs))
    self.assertTrue(self.absallclose(actual_eigvals, expected_eigvals))


if __name__ == '__main__':
  unittest.main()