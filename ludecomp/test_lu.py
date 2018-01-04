import unittest
import numpy as np
import scipy.linalg as LA

from lu import LU
from utils import multi_dot


class LUDecompositionTest(unittest.TestCase):
    """
    Tests LU decomposition with various pivoting strategies.
    """

    def test_no_pivoting(self):
        T = np.array([
            [2, 1, 1, 0],
            [4, 3, 3, 1],
            [8, 7, 9, 5],
            [6, 7, 9, 8]
        ])

        L_a, U_a = LU()(T)
        actual = np.dot(L_a, U_a)
        self.assertTrue(np.allclose(actual, T))

    def test_partial_pivoting(self):
        T = np.array([
            [2, 1, 1, 0],
            [4, 3, 3, 1],
            [8, 7, 9, 5],
            [6, 7, 9, 8]
        ])

        actual = LU(pivoting='partial')(T)
        expected = LA.lu(T)

        self.assertTrue(np.allclose(a, e) for a, e in zip(actual, expected))

    def test_full_pivoting(self):
        T = np.array([
            [2, 1, 1, 0],
            [4, 3, 3, 1],
            [8, 7, 9, 5],
            [6, 7, 9, 8]
        ])

        actual = list(LU(pivoting='full')(T))
        self.assertTrue(np.allclose(multi_dot(actual), T))


if __name__ == '__main__':
    unittest.main()
