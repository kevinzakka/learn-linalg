import numpy as np

from linalg.eigen.single import power_iteration, inverse_iteration


if __name__ == "__main__":
  M = np.random.randn(3, 3)
  M = M.T @ M

  eigvals, eigvecs = np.linalg.eig(M)
  idx = eigvals.argsort()[::-1]
  eigvecs = eigvecs[:, idx]
  expected_big = eigvecs[:, 0]
  expected_small = eigvecs[:, -1]

  actual_big = power_iteration(M)
  actual_small = inverse_iteration(M)

  assert np.allclose(actual_big, expected_big)
  assert np.allclose(actual_small, expected_small)
