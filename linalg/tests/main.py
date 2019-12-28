import time
import numpy as np
import numpy.linalg as LA

from scipy.linalg import hessenberg
from linalg import utils
from linalg.svd import SVD
from linalg.eigen import single, multi

np.set_printoptions(precision=4)


if __name__ == "__main__":
  A = np.random.randn(7, 5)

  hess = hessenberg(A)

  # U, S, V = SVD(A).decompose()
  # assert np.allclose(U@S@V.T, A)