import unittest
import numpy as np
import numpy.linalg as LA

from linalg.qrdecomp.qr import QR
from linalg.utils import diag, create_diag


class QRTest(unittest.TestCase):
  """Tests QR decomposition with various strategies.
  """
  def test_gram_schmidt(self):
    T = np.random.randn(100, 60)

    actual = QR(T).gram_schmidt()
    Q, R = LA.qr(T)

    # enforce uniqueness for numpy version
    D = create_diag(np.sign(diag(R)))
    Q = np.dot(Q, D)
    R = np.dot(D, R)
    expected = (Q, R)

    self.assertTrue(all(np.allclose(a, e) for a, e in zip(actual, expected)))

  def test_householder_complete(self):
    T = np.random.randn(100, 60)

    actual = QR(T, reduce=False).householder()
    expected = LA.qr(T, mode='complete')

    self.assertTrue(all(np.allclose(a, e) for a, e in zip(actual, expected)))

  def test_householder_reduce(self):
    T = np.random.randn(100, 60)

    actual = QR(T, reduce=True).householder()
    expected = LA.qr(T, mode='reduced')

    self.assertTrue(all(np.allclose(a, e) for a, e in zip(actual, expected)))

  def test_solve_single_complete(self):
    A = np.random.randn(100, 60)
    b = np.random.randn(100)

    actual = QR(A, reduce=False).solve(b)
    expected = LA.lstsq(A, b, rcond=None)[0]

    self.assertTrue(np.allclose(actual, expected))

  def test_solve_single_reduce(self):
    A = np.random.randn(100, 60)
    b = np.random.randn(100)

    actual = QR(A, reduce=True).solve(b)
    expected = LA.lstsq(A, b, rcond=None)[0]

    self.assertTrue(np.allclose(actual, expected))

  def test_solve_multi_complete(self):
    A = np.random.randn(100, 60)
    b = np.random.randn(100, 10)

    actual = QR(A, reduce=False).solve(b)
    expected = LA.lstsq(A, b, rcond=None)[0]

    self.assertTrue(np.allclose(actual, expected))

  def test_solve_multi_reduce(self):
    A = np.random.randn(100, 60)
    b = np.random.randn(100, 10)

    actual = QR(A, reduce=True).solve(b)
    expected = LA.lstsq(A, b, rcond=None)[0]

    self.assertTrue(np.allclose(actual, expected))


if __name__ == '__main__':
  unittest.main()