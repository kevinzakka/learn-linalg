import unittest
import numpy as np
import numpy.linalg as LA

from solver.solve import solve


class SolveTest(unittest.TestCase):
    """
    Tests solve for a well determined system.
    """

    def test_well_determined_partial(self):
        A = np.random.randn(50, 50)
        b = np.random.randn(50)

        actual = solve(A, b, 'partial')
        expected = LA.solve(A, b)

        self.assertTrue(np.allclose(actual, expected))

    def test_well_determined_full(self):
        A = np.random.randn(50, 50)
        b = np.random.randn(50)

        actual = solve(A, b, 'full')
        expected = LA.solve(A, b)

        self.assertTrue(np.allclose(actual, expected))


if __name__ == '__main__':
    unittest.main()
