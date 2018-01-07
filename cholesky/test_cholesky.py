import unittest
import numpy as np
import scipy.linalg as LA

from cholesky import Cholesky


class CholeskyDecompositionTest(unittest.TestCase):
    """
    Tests Cholesky decomposition.
    """

    def test_decompose(self):
        T = np.array([
            [4, 12, -16],
            [12, 37, -43],
            [-16, -43, 98]
        ])

        actual = Cholesky(T).decompose()
        expected = LA.cholesky(T)

        self.assertTrue(np.allclose(actual, expected))


if __name__ == '__main__':
    unittest.main()
