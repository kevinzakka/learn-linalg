import numpy as np
import numpy.linalg as LA

from linalg.eigen import multi
from linalg.utils import random_symmetric


if __name__ == "__main__":
  N = 3
  M = random_symmetric(N)

  expected_eigvals, expected_eigvecs = LA.eig(M)

  actual_eigvals, actual_eigvecs = multi.projected_iteration(M, N, 10000)

  print(actual_eigvals)
  print(expected_eigvals)