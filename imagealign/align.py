import pickle
import numpy as np
import matplotlib.pyplot as plt

from cholesky import Cholesky
from imutils import load_coords
from app import InteractiveImage


def main():

    img_dir = './imgs/'
    size = (800, 800)
    inter = InteractiveImage(img_dir, size)
    inter.show()


if __name__ == '__main__':
    main()
