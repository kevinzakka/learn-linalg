import unittest
import numpy as np

from linalg.kahan import kahan_sum


class KahanSumTest(unittest.TestCase):
  """Tests Kahan summation.
  """
  def test_sum(self):
    x = np.random.randn(int(1e5))
    expected = x.sum()
    actual = kahan_sum(x)
    self.assertTrue(np.allclose(expected, actual))

  def test_sum_axis_zeroth(self):
    x = np.random.randn(100, 100)
    expected = x.sum(axis=0)
    actual = kahan_sum(x, axis=0)
    self.assertTrue(np.allclose(expected, actual))

  def test_sum_axis_zeroth_keepdims(self):
    x = np.random.randn(100, 100)
    expected = x.sum(axis=0, keepdims=True)
    actual = kahan_sum(x, axis=0, keepdims=True)
    self.assertTrue(np.allclose(expected, actual))

  def test_sum_axis_first(self):
    x = np.random.randn(100, 100)
    expected = x.sum(axis=1)
    actual = kahan_sum(x, axis=1)
    self.assertTrue(np.allclose(expected, actual))

  def test_sum_axis_first_keepdims(self):
    x = np.random.randn(100, 100)
    expected = x.sum(axis=1, keepdims=True)
    actual = kahan_sum(x, axis=1, keepdims=True)
    self.assertTrue(np.allclose(expected, actual))


if __name__ == '__main__':
  unittest.main()