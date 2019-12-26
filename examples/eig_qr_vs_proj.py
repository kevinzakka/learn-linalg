"""QR algorithm vs projected iteration.
"""

import time

import numpy as np
import numpy.linalg as LA

from linalg.eigen import multi
from linalg import utils


if __name__ == "__main__":
  M = utils.random_spd(10)

  tic = time.time()
  eigvals, eigvecs = multi.qr_algorithm(M, hess=False)
  toc = time.time()
  time_qr = toc - tic
  print("QR: {}s".format(time_qr))

  tic = time.time()
  eigvals, eigvecs = multi.projected_iteration(M, len(M))
  toc = time.time()
  time_proj = toc - tic
  print("Projected Iteration: {}s".format(time_proj))