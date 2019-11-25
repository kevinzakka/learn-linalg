import numpy as np

from linalg.eigen.single import *


def same_eigvec(x, y):
  if np.sign(x[0]) == np.sign(y[0]):
    diff = x - y
  else:
    diff = x + y
  return np.allclose(diff, np.zeros(3))


if __name__ == "__main__":
  M = np.array([[1, 7, 3], [7, 4, -5], [3, -5, 6]])

  eigvals, eigvecs = np.linalg.eig(M)
  idx = eigvals.argsort()[::-1]
  eigvecs = eigvecs[:, idx]
  eigvals = eigvals[idx]
  expected_big = eigvecs[:, 0]
  expected_small = eigvecs[:, -1]

  actual_big = power_iteration(M, 10000)
  actual_small = inverse_iteration(M, 10000)

  assert(same_eigvec(actual_big, expected_big))
  assert(same_eigvec(actual_small, expected_small))

  expected_eigval = eigvals[0]
  actual_eigval = rayleigh_quotient(M, actual_big)

  print("expected: ", expected_eigval)
  print("actual: ", actual_eigval)
