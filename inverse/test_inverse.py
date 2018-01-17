import unittest
import numpy as np
import numpy.linalg as LA

from inverse import inverse


class InverseTest(unittest.TestCase):
    """
    Tests the inverse of a square matrix A.
    """

    def test_inverse_simple(self):
        T = np.array([
            [2, 1, 1, 0],
            [4, 3, 3, 1],
            [8, 7, 9, 5],
            [6, 7, 9, 8]
        ])

        actual = inverse(T)
        expected = LA.inv(T)

        self.assertTrue(np.allclose(actual, expected))

    def test_inverse_random(self):
        T = np.random.randn(100, 100)

        actual = inverse(T)
        expected = LA.inv(T)

        self.assertTrue(np.allclose(actual, expected))


if __name__ == '__main__':
    unittest.main()
