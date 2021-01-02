import numpy as np

from linalg.kahan import KahanSum
from linalg.utils import upper_diag
from linalg.eigen.utils import is_spd


class Cholesky:
  """Computes the Cholesky factorization of an SPD matrix A.

  Instead of seeking arbitrary lower and upper triangular
  factors L and U, the Cholesky factorization exploits the
  symmetry properties of A to construct a lower triangular
  matrix L whose transpose can serve as the upper triangular
  part. When used on SPD matrices, Cholesky will run twice as
  fast as LU.

  Cholesky factorization analogously works on the complex
  equivalent of symmetric matrices, Hermitian matrices, in
  which case it returns L and its conjugate transpose L.H.

  Note that the decomposition will fail if the matrix is not
  symmetric or Hermitian.

  Args:
    A: a numpy array of shape (N, N).
    crout: Whether to use the Crout algorithm. From my
        experience, setting it to False yields a faster
        factorization.

  Returns:
    L: a lower triangular numpy array of shape (N, N).
  """
  def __init__(self, A, crout=False):
    error_msg = 'A must be symmetric positive definite.!'
    assert is_spd(A), error_msg

    self.crout = crout
    if self.crout:
      self.R = np.array(A)
    else:
      self.R = upper_diag(np.array(A), diag=True)

  def decompose(self, ret=True):
    N = len(self.R)

    self.L = np.zeros_like(self.R)

    if self.crout:
      for i in range(N):
        for j in range(i+1):
          summer = KahanSum()
          for k in range(j):
            summer.add(self.L[i, k] * self.L[j, k])
          sum = summer.result()

          if (i == j):
            self.L[j, j] = np.sqrt(self.R[j, j] - sum)
          else:
            self.L[i, j] = (self.R[i, j] - sum) / (self.L[j, j])
      if ret:
        return self.L
    else:
      for i in range(N):
        self.pivot = self.R[i, i]

        # eliminate subsequent rows
        for j in range(i+1, N):
          for k in range(j, N):
            self.R[j, k] -= self.R[i, k] * \
                    (self.R[i, j] / self.pivot)

        # scale the current row
        for k in range(i, N):
          self.R[i, k] /= np.sqrt(self.pivot)

      if ret:
        return self.R.T

  def solve(self, b):
    """Solves the lineary system `Ax = b`.

    Concretely, performs the Cholesky factorization of
    the matrix A, then solves the system using forward
    and backward substitution.
    """
    self.b = b

    self.decompose(ret=False)

    if not self.crout:
      self.L = self.R.T

    self._forward()
    self._backward()

    return self.x

  def _forward(self):
    """Forward substituion.

    Solves the lower triangular system `Ly = b` for y
    by forward substitution.
    """
    if self.b.ndim > 1:
      num_iters = self.b.shape[1]
      N = self.b.shape[0]
    else:
      num_iters = 1
      N = self.b.shape[0]

    self.y = np.zeros([N, num_iters])

    for k in range(num_iters):
      for i in range(N):
        acc = KahanSum()
        for j in range(i):
          acc.add(self.L[i, j]*self.y[j, k])
        if self.b.ndim > 1:
          self.y[i, k] = \
            (self.b[i, k] - acc.result()) / (self.L[i, i])
        else:
          self.y[i, k] = \
            (self.b[i] - acc.result()) / (self.L[i, i])

  def _backward(self):
    """Backward substitution.

    Solves the upper triangular system `L^Tx = y` for x
    by back substitution.
    """
    if self.b.ndim > 1:
      num_iters = self.b.shape[1]
      N = self.b.shape[0]
    else:
      num_iters = 1
      N = self.b.shape[0]

    self.x = np.zeros([N, num_iters])

    for k in range(num_iters):
      for i in range(N-1, -1, -1):
        acc = KahanSum()
        for j in range(N-1, i, -1):
          acc.add(self.L.T[i, j]*self.x[j, k])
        self.x[i, k] = \
          (self.y[i, k] - acc.result()) / (self.L.T[i, i])

    if self.b.ndim == 1:
      self.x = self.x.squeeze()
