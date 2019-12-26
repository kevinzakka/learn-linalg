import time
import numpy as np
import numpy.linalg as LA

from linalg.eigen.multi import qr_algorithm
from linalg import utils
np.set_printoptions(precision=3)


if __name__ == "__main__":
  M = utils.random_spd(3)

  actual_eigvals, actual_eigvecs = qr_algorithm(M)

  eigvals, eigvecs = LA.eig(M)
  idx = np.abs(eigvals).argsort()[::-1]
  expected_eigvecs = eigvecs[:, idx]
  expected_eigvals = eigvals[idx]

  print(expected_eigvecs)
  print(actual_eigvecs)