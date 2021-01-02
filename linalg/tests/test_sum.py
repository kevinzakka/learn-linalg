import unittest
import itertools
import numpy as np

from linalg.kahan import kahan_sum

# All possible axis and keepdims combinations.
param_list = itertools.product([0, 1, None], [False, True])


class KahanSumTest(unittest.TestCase):
  """Tests Kahan summation."""

  def test_sum_1D(self):
    x = np.random.randn(int(1e5))
    expected = x.sum()
    actual = kahan_sum(x)
    self.assertTrue(np.allclose(expected, actual))

  def test_sum_2D(self):
    x = np.random.randn(100, 100)
    for ax, kd in param_list:
      with self.subTest(axis=ax, keepdims=kd):
        expected = x.sum(axis=ax, keepdims=kd)
        actual = kahan_sum(x, axis=ax, keepdims=kd)
        self.assertTrue(np.allclose(expected, actual))


if __name__ == '__main__':
  unittest.main()
