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

        L_a, U_a = LU(T).decompose()
        actual = np.dot(L_a, U_a)

        self.assertTrue(np.allclose(actual, T))

    def test_partial_pivoting(self):
        T = np.array([
            [2, 1, 1, 0],
            [4, 3, 3, 1],
            [8, 7, 9, 5],
            [6, 7, 9, 8]
        ])

        actual = LU(T, pivoting='partial').decompose()
        expected = LA.lu(T)

        self.assertTrue(np.allclose(a, e) for a, e in zip(actual, expected))

    def test_full_pivoting(self):
        T = np.array([
            [2, 1, 1, 0],
            [4, 3, 3, 1],
            [8, 7, 9, 5],
            [6, 7, 9, 8]
        ])

        actual = list(LU(T, pivoting='full').decompose())
        actual[0] = actual[0].T

        self.assertTrue(np.allclose(multi_dot(actual), T))

    def test_solve_single_no_pivoting(self):
        T = np.array([
            [2, 1, 1, 0],
            [4, 3, 3, 1],
            [8, 7, 9, 5],
            [6, 7, 9, 8]
        ])
        b = np.array([1, 4, 5, 2])

        actual = LU(T).solve(b)
        expected = np.linalg.solve(T, b)

        self.assertTrue(np.allclose(actual, expected))

    def test_solve_multi_no_pivoting(self):
        T = np.array([
            [2, 1, 1, 0],
            [4, 3, 3, 1],
            [8, 7, 9, 5],
            [6, 7, 9, 8]
        ])
        b = np.array([[1, 4, 5, 2], [2, 1, 2, 1]]).T

        actual = LU(T).solve(b)
        expected = np.linalg.solve(T, b)

        self.assertTrue(np.allclose(actual, expected))

    def test_solve_multi_partial_pivoting(self):
        T = np.array([
            [2, 1, 1, 0],
            [4, 3, 3, 1],
            [8, 7, 9, 5],
            [6, 7, 9, 8]
        ])
        b = np.array([[1, 4, 5, 2], [2, 1, 2, 1]]).T

        actual = LU(T, pivoting='partial').solve(b)
        expected = np.linalg.solve(T, b)

        self.assertTrue(np.allclose(actual, expected))

    def test_solve_multi_full_pivoting(self):
        T = np.array([
            [2, 1, 1, 0],
            [4, 3, 3, 1],
            [8, 7, 9, 5],
            [6, 7, 9, 8]
        ])
        b = np.array([[1, 4, 5, 2], [2, 1, 2, 1]]).T

        actual = LU(T, pivoting='full').solve(b)
        expected = np.linalg.solve(T, b)

        self.assertTrue(np.allclose(actual, expected))


if __name__ == '__main__':
    unittest.main()
