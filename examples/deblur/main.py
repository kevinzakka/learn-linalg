"""Deblur a blurred image.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from blur import GaussianFilter


def main(args):
  img_name = os.path.join("./imgs/", args.name)
  img = np.asarray(Image.open(img_name))

  blur_filter = GaussianFilter(2)
  img_blurred = blur_filter.filter(img)

  plt.imshow(img_blurred)
  plt.show()



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("name", type=str,
                      help="Name of the image in the img folder.")
  args, unparsed = parser.parse_known_args()
  main(args)
