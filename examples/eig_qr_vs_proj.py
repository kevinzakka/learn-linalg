"""This script illustrates how computing the Hessenberg
form of a matrix can help gain a speedup in computing
its eigenpairs using the QR algorithm.
"""

import time

import numpy as np
import numpy.linalg as LA

from linalg.eigen import multi
from linalg import utils


if __name__ == "__main__":
  M = utils.random_spd(30)

  tic = time.time()
  eigvals, eigvecs = multi.qr_algorithm(M, hess=True)
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