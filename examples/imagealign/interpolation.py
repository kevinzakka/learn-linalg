import numpy as np


def affine_grid_generator(height, width, M):
  """This function returns a sampling grid, which when
  used with the bilinear sampler on the input img,
  will create an output img that is an affine
  transformation of the input.

  Args:
    M: affine transform matrices of shape (2, 3).
    height: height of the input img.
    width: width of the input img.

  Returns:
    normalized grid (-1, 1) of shape (H, W, 2).
    The 3rd dimension has 2 components: (x, y) which are the
    sampling points of the original image for each point in
    the target image.
  """
  # create normalized 2D grid
  x = np.linspace(-1, 1, width)
  y = np.linspace(-1, 1, height)
  x_t, y_t = np.meshgrid(x, y)

  # convert to homogeneous coordinates
  ones = np.ones(np.prod(x_t.shape))
  sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

  # affine transform sampling grid
  affine_grid = np.dot(M, sampling_grid)
  # affine grid has shape (2, H*W)

  # reshape to (H, W, 2)
  affine_grid = affine_grid.reshape(2, height, width)

  return affine_grid


def bilinear_sampler(input_img, x, y):
  """Bilinear interpolation.

  Performs bilinear sampling of the input img according to the
  normalized coordinates provided by the sampling grid. Note that
  the sampling is done identically for each channel of the input.

  To test if the function works properly, output image should be
  identical to input image when M is initialized to identity
  transform.

  Args:
    input_img: numpy array of shape (H, W, C).
    grid: x, y which is the output of affine_grid_generator.

  Returns:
    interpolated img according to grid.
  """
  # grab dimensions
  H, W, C = input_img.shape

  # rescale x and y to [0, W/H]
  x = ((x + 1.) * W) * 0.5
  y = ((y + 1.) * H) * 0.5

  # grab 4 nearest corner points for each (x_i, y_i)
  x0 = np.floor(x).astype(np.int64)
  x1 = x0 + 1
  y0 = np.floor(y).astype(np.int64)
  y1 = y0 + 1

  # make sure it's inside img range [0, H] or [0, W]
  x0 = np.clip(x0, 0, W-1)
  x1 = np.clip(x1, 0, W-1)
  y0 = np.clip(y0, 0, H-1)
  y1 = np.clip(y1, 0, H-1)

  # look up pixel values at corner coords
  Ia = input_img[y0, x0]
  Ib = input_img[y1, x0]
  Ic = input_img[y0, x1]
  Id = input_img[y1, x1]

  # calculate deltas
  wa = (x1-x) * (y1-y)
  wb = (x1-x) * (y-y0)
  wc = (x-x0) * (y1-y)
  wd = (x-x0) * (y-y0)

  # add dimension for addition
  wa = np.expand_dims(wa, axis=2)
  wb = np.expand_dims(wb, axis=2)
  wc = np.expand_dims(wc, axis=2)
  wd = np.expand_dims(wd, axis=2)

  # compute output
  out = wa*Ia + wb*Ib + wc*Ic + wd*Id

  return out