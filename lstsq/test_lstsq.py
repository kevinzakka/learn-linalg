import unittest
import numpy as np
import numpy.linalg as LA

from lstsq import lstsq


class LeastSquaresTest(unittest.TestCase):
    """
    Tests Linear Least Squares for an over
    or well determined system.
    """

    def test_well_determined(self):
        A = np.random.randn(10, 10)
        b = np.random.randn(10)

        actual = lstsq(A, b)
        expected = LA.lstsq(A, b)[0]

        self.assertTrue(np.allclose(actual, expected))

    def test_over_determined(self):
        A = np.random.randn(20, 15)
        b = np.random.randn(20)

        actual = lstsq(A, b)
        expected = LA.lstsq(A, b)[0]

        self.assertTrue(np.allclose(actual, expected))


if __name__ == '__main__':
    unittest.main()
