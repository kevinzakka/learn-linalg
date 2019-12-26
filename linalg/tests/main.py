import time
import numpy as np
import numpy.linalg as LA

from linalg import utils
from linalg.eigen import single
np.set_printoptions(precision=3)


if __name__ == "__main__":
  M = utils.random_symmetric(3)
  actual_eigval, actual_eigvec = single.power_iteration(M, 1000)

  eigvals, eigvecs = LA.eig(M)
  # sort by largest absolute eigenvalue
  idx = np.abs(eigvals).argsort()[::-1]
  eigvecs = eigvecs[:, idx]
  eigvals = eigvals[idx]
  expected_eigval, expected_eigvec = eigvals[0], eigvecs[:, 0]

  print(actual_eigvec)
  print(expected_eigvec)