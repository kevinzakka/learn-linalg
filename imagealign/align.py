import pickle
import numpy as np
import matplotlib.pyplot as plt

from app import InteractiveImage
from cholesky import Cholesky


def process(coords):
    """
    Process the coordinates as:

    - y: the coordinates of the crooked image
    - x: the coordinates of the reference image
    """
    x = [[l[0][0], l[0][1]] for l in coords]
    y = [[l[1][0], l[1][1]] for l in coords]

    x = np.array(x)
    y = np.array(y)

    return x, y


def main():

    img_dir = './imgs/'
    inter = InteractiveImage(img_dir)
    inter.show()

    # read the pickle dump and process
    coords = pickle.load(open("./dump/coords.p", "rb"))
    x, y = process(coords)

    # homogenize coordinates
    y = np.append(y, np.ones((y.shape[0],1)), axis=1)

    # solve normal equation with cholesky
    M = Cholesky(np.dot(y.T, y)).solve(np.dot(y.T, x))

    # 

if __name__ == '__main__':
    main()
