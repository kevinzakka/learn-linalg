"""If a linear algebra problem is hard, substitute the SVD.
"""

import numpy as np

from linalg.eigen.multi import qr_algorithm
from linalg.qrdecomp import QR
from linalg.utils import create_diag


class SVD:
  """Computes the Singular Value Decomposition of an `m x n` matrix A.

  The matrix A is factorized into a product of three matrices: (1) an
  orthogonal matrix U, (2) a diagonal matrix Σ and (3) an orthogonal
  matrix V, such that `A = U x Σ x V^T`.

  The SVD decomposition exists for any matrix, even if it is not
  symmetric or square.
  """
  def __init__(self, A):
    """Constructor.

    Args:
      A: A 2D array of shape (M, N).
    """
    assert A.ndim == 2, "[!] A must be 2-D."
    self.backup = np.array(A, dtype=np.float64)

  def decompose(self):
    """Compute the SVD in 2 steps:

    1. Compute the eigenvalue decomposition of `A.T @ A`
      using the QR algorithm. From this decomposition,
      we can derive V and and Σ.
    2. Solve the system `UΣ = AV` for U using a QR
      factorization.

    This algorithm isn't usually used in practice
    because of its sensitivity to the gram matrix
    `A.T @ A`.

    Returns:
      U: A numpy array of shape (M, M)
      S: A numpy array of shape (M, N).
      V: A numpy array of shape (N, N).
    """
    A = np.array(self.backup)
    M, N = A.shape
    eigvals, V = qr_algorithm(A.T @ A)
    S = create_diag(np.sqrt(eigvals), (M, N))
    U = QR(S.T, reduce=False).solve(V.T @ A.T)
    U = U.T
    return U, S, V