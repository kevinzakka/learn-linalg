import os
import numpy as np
import matplotlib.pyplot as plt

from cholesky import Cholesky
from app import InteractiveImage
from imutils import img2array, load_coords, homogenize
from interpolation import affine_grid_generator, bilinear_sampler


def main():

    img_dir = './imgs/'
    H, W = 800, 800
    inter = InteractiveImage(img_dir, (H, W))
    inter.show()

    # load and homogenize coordinates
    x, y = load_coords('coords.p', H, W)
    y = homogenize(y)

    # create P array
    num_pts = y.shape[0]

    P = []
    for i in range(num_pts):
        P.append(np.hstack([y[i], np.zeros(3).T]))
        P.append(np.hstack([np.zeros(3).T, y[i]]))
    P = np.array(P)

    # create Q array
    Q = x.flatten()

    # solve for M using Cholesky factorization
    M = Cholesky(np.dot(P.T, P)).solve(np.dot(P.T, Q))
    M = M.reshape(2, 3)
    M[np.abs(M) < 1e-10] = 0

    # read crooked image
    included_extensions = ['png']
    file_names = [
        fn for fn in os.listdir(img_dir)
        if any(fn.endswith(ext) for ext in included_extensions)
        if "crooked" in fn
    ]
    img = img2array(img_dir + file_names[0], desired_size=(H, W))

    # affine transform and interpolate
    affine_grid = affine_grid_generator(H, W, M)
    x_s = affine_grid[0:1, :].squeeze()
    y_s = affine_grid[1:2, :].squeeze()
    out = bilinear_sampler(img, x_s, y_s)
    out = np.clip(out, 0, 1)

    imgs = np.array([img, out])
    N = imgs.shape[0]

    # plot
    fig, ax = plt.subplots(nrows=1, ncols=N)
    titles = ['Crooked', "Fixed"]

    for i in range(N):
        ax[i].imshow(imgs[i])
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        ax[i].set_title(titles[i])

    plt.show()


if __name__ == '__main__':
    main()
