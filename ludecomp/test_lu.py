import unittest
import numpy as np
import scipy.linalg as LA

from lu import LU


class LUDecompositionTest(unittest.TestCase):
    """
    Tests LU decomposition with various pivoting strategies.
    """

    def test_no_pivoting(self):
        T = A = np.array([
            [2, 1, 1, 0], 
            [4, 3, 3, 1], 
            [8, 7, 9, 5], 
            [6, 7, 9, 8]
        ])

        expected = LA.lu(T)
        actual = LU()(T)

        self.assertTrue(np.allclose(e, a) for e,a in zip(expected, actual))