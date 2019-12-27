import time
import numpy as np
import numpy.linalg as LA

from scipy.linalg import hessenberg
from linalg import utils
from linalg.qrdecomp import QR
from linalg.eigen import single, multi

np.set_printoptions(precision=3)


if __name__ == "__main__":
  A = utils.random_symmetric(5)

  tic = time.time()
  hess = multi.hessenberg(A)
  toc = time.time()
  print(hess)
  assert utils.is_symm(hess), "[!] Hessenberg of symmetric isn't symmetric."