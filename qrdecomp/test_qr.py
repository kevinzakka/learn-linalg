import unittest
import numpy as np
import numpy.linalg as LA

from qr import QR
from utils import diag, create_diag


class QRTest(unittest.TestCase):
    """
    Tests QR decomposition with various strategies.
    """

    def test_gram_schmidt(self):
        # T = np.random.randn(100, 100)
        # T = np.array([[1., 4, -2], [2, 5, 6], [7, 0, 3]], dtype=np.float64)
        T = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float64)

        actual = QR(T).gram_schmidt()
        Q, R = LA.qr(T)

        # enforce uniqueness for numpy version
        D = create_diag(np.sign(diag(R)))
        Q = np.dot(Q, D)
        R = np.dot(D, R)
        expected = (Q, R)

        self.assertTrue(all(np.allclose(a, e) for a, e in zip(actual, expected)))


if __name__ == '__main__':
    unittest.main()
