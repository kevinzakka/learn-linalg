import numpy as np

from linalg.eigen.multi import hessenberg


if __name__ == "__main__":
  M = np.random.randint(0, 4, size=(5, 5))

  hess = hessenberg(M)

  print(M)