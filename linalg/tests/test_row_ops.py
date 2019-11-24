import unittest
import numpy as np
import gelim.row_ops as row_ops

from utils.utils import multi_dot


class RowOpsTest(unittest.TestCase):
    """
    Tests elementary row operations.
    """

    def test_permute_full(self):
        T = np.random.randn(3, 3)
        idx = [2, 0, 1]

        P = row_ops.permute(T.shape[0], idx)
        actual = np.dot(P, T)

        expected = T[idx, :]

        self.assertTrue(np.allclose(expected, actual))

    def test_permute_no_full(self):
        T = np.random.randn(8, 8)
        idx = [(0, 2)]

        P = row_ops.permute(T.shape[0], idx)
        actual = np.dot(P, T)

        expected = T[[2, 1, 0, 3, 4, 5, 6, 7], :]

        self.assertTrue(np.allclose(expected, actual))

    def test_permute_throws_exception(self):
        T = np.random.randn(3, 3)
        idx = [(3, 1)]
        self.assertRaises(AssertionError, row_ops.permute, T.shape[0], idx)
        idx = [0, 3, 1]
        self.assertRaises(AssertionError, row_ops.permute, T.shape[0], idx)

    def test_scale_full(self):
        T = np.random.randn(3, 3)
        scalars = [1, 1, 0.5]

        S = row_ops.scale(T.shape[0], scalars)
        actual = np.dot(S, T)

        scalars = np.array(scalars)
        expected = T * scalars[:, np.newaxis]

        self.assertTrue(np.allclose(expected, actual))

    def test_scale_no_full(self):
        T = np.random.randn(3, 3)
        scalars = [(2, 0.5)]

        S = row_ops.scale(T.shape[0], scalars)
        actual = np.dot(S, T)

        scalars = np.array([1, 1, 0.5])
        expected = T * scalars[:, np.newaxis]

        self.assertTrue(np.allclose(expected, actual))

    def test_eliminate(self):
        T = np.random.randn(3, 3)
        k, c, l = 0, 5, 2

        E = row_ops.eliminate(T.shape[0], k, c, l)
        actual = np.dot(E, T)

        expected = T
        expected[l] += c*T[k]
        self.assertTrue(np.allclose(expected, actual))

    def test_cascade(self):
        T = np.array([[0, 1, -1], [3, -1, 1], [1, 1, -2]])
        N = T.shape[0]

        transformations = []
        transformations.append(row_ops.permute(N, [2, 0, 1]))
        transformations.append(row_ops.eliminate(N, 0, -3, 2))
        transformations.append(row_ops.eliminate(N, 1, 4, 2))
        transformations.append(row_ops.scale(N, [(2, 1/3)]))
        transformations.append(row_ops.eliminate(N, 2, 2, 0))
        transformations.append(row_ops.eliminate(N, 2, 1, 1))
        transformations.append(row_ops.eliminate(N, 1, -1, 0))
        transformations = transformations[::-1]

        actual = multi_dot(transformations)
        expected = np.linalg.inv(T)
        self.assertTrue(np.allclose(expected, actual))


if __name__ == '__main__':
    unittest.main()
