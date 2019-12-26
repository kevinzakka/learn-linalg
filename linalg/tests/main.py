import time
import numpy as np
import numpy.linalg as LA

from linalg.eigen.multi import hessenberg
from linalg.qrdecomp import QR
np.set_printoptions(precision=3)


if __name__ == "__main__":
  M = np.random.randint(0, 10, size=(5, 5)).astype("float32")
  print(M)
  hess, vs = hessenberg(M)
  print(hess)