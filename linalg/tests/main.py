import time
import numpy as np
import numpy.linalg as LA

from linalg import utils
from linalg.svd import SVD

np.set_printoptions(precision=3)


if __name__ == "__main__":
  A = np.random.randn(7, 5)
  # A = utils.random_symmetric(3)
  U, S, V = SVD(A).decompose()
  assert np.allclose(U@S@V.T, A)