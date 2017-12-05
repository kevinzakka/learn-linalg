import unittest
import numpy as np

from gaussian_elim import GaussElim


class GaussianElimTest(unittest.TestCase):
    """
    Test Gaussian Elimination for solving linear
    systems of equations.
    """

    def test_no_pivoting(self):
        A = np.array([[3, 1], [1, 2]])
        b = np.array([9, 8])

        expected = np.linalg.solve(A, b)

        gelim = GaussElim(A, b)
        actual = gelim.solve()[0]

        self.assertTrue(np.allclose(expected, actual))

    def test_pivoting_partial(self):
        A = np.array([[1, 1, 1], [2, 1, 3], [3, 1, 6]])
        b = b = np.array([4, 7, 2])

        expected = np.linalg.solve(A, b)

        gelim = GaussElim(A, b, pivoting='partial')
        actual = gelim.solve()[0]

        self.assertTrue(np.allclose(expected, actual))

    def test_pivoting_full(self):
        A = np.array([[3, 1], [1, 2]])
        b = b = np.array([9, 8])

        expected = np.linalg.solve(A, b)

        gelim = GaussElim(A, b, pivoting='full')
        actual = gelim.solve()[0]

        self.assertTrue(np.allclose(expected, actual))


if __name__ == '__main__':
    unittest.main()
