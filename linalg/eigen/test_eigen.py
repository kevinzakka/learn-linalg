import numpy as np

from linalg.eigen.single import *


def same_eigvec(x, y):
  if np.sign(x[0]) == np.sign(y[0]):
    diff = x - y
  else:
    diff = x + y
  return np.allclose(diff, np.zeros(3))


if __name__ == "__main__":
  M = np.random.randn(3, 3)
  M = M.T @ M

  eigvals, eigvecs = np.linalg.eig(M)
  idx = eigvals.argsort()[::-1]
  eigvecs = eigvecs[:, idx]
  eigvals = eigvals[idx]
  expected_eigvec_big = eigvecs[:, 0]
  expected_eigvec_small = eigvecs[:, -1]
  expected_eigval_big = eigvals[0]
  expected_eigval_small = eigvals[-1]

  actual_eigval_big, actual_eigvec_big = power_iteration(M, 10000)
  actual_eigval_small, actual_eigvec_small = inverse_iteration(M, 10000)

  assert(same_eigvec(actual_eigvec_big, expected_eigvec_big))
  assert(same_eigvec(actual_eigvec_small, expected_eigvec_small))

  assert(np.isclose(actual_eigval_big, expected_eigval_big))
  assert(np.isclose(actual_eigval_small, expected_eigval_small))
  
  # test rayleigh quotient iteration for largest
  ray_eigval, ray_eigvec = rayleigh_quotient_iteration(M, power_iteration(M, 50)[0])
  assert(same_eigvec(ray_eigvec, expected_eigvec_big))

  # test rayleigh quotient iteration for smallest
  ray_eigval, ray_eigvec = rayleigh_quotient_iteration(M, inverse_iteration(M, 50)[0])
  assert(same_eigvec(ray_eigvec, expected_eigvec_small))
