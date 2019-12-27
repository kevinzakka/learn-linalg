"""This script illustrates how computing the Hessenberg
form of a matrix can help gain a speedup in computing
its eigenpairs using the QR algorithm.

Note: Currently not getting any speedups because of how
my code uses a mix of for loops and numpy vectorization.
Specifically, the QR decomposition doesn't take advantage
of the Hessenberg form by skipping row elements that are
already zero when applying reflectors to each column
of the matrix.
"""

import time

import numpy as np
import scipy.linalg as LA

from linalg.eigen import multi
from linalg import utils


if __name__ == "__main__":
  M = utils.random_spd(10)

  tic = time.time()
  eigvals, eigvecs = multi.qr_algorithm(LA.hessenberg(M), hess=False)
  toc = time.time()
  time_with_hess = toc - tic
  print("With hessenberg: {}s".format(time_with_hess))

  tic = time.time()
  eigvals, eigvecs = multi.qr_algorithm(M, hess=False)
  toc = time.time()
  time_without_hess = toc - tic
  print("Without hessenberg: {}s".format(time_without_hess))

  speedup = time_without_hess / time_with_hess
  print("Speedup: {}x".format(speedup))