import unittest
import numpy as np
import numpy.linalg as LA

from qr import QR


class QRTest(unittest.TestCase):
    """
    Tests QR decomposition with various strategies.
    """

    def test_gram_schmidt(self):
        T = np.random.randn(100, 100)

        actual = QR(T).gram_schmidt()
        expected = LA.qr(T)

        self.assertTrue(np.allclose(a, e) for a, e in zip(actual, expected))


if __name__ == '__main__':
    unittest.main()
