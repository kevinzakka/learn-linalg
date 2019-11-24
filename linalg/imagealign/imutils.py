import pickle
import numpy as np

from PIL import Image


def homogenize(x):
    """
    Homogenize a vector of coordinates,
    i.e. add a column of ones to a 2D
    (x, y) tuple to make it 3D.
    """
    ones = np.ones((x.shape[0], 1), dtype=x.dtype)
    return np.append(x, ones, axis=1)


def load_coords(filename, H, W):
    """
    Load coordinate pickle dump and return:

    - y: the coordinates of the crooked image
    - x: the coordinates of the reference image
    """
    dump_dir = './dump/' + filename
    coords = pickle.load(open(dump_dir, "rb"))

    x = [[l[0][0], l[0][1]] for l in coords]
    y = [[l[1][0], l[1][1]] for l in coords]

    x = np.array(x)
    y = np.array(y)

    x = ((2*x) / W) - 1
    y = ((2*y) / H) - 1

    return x, y


def resize_array(x, size):
    # 3D and 4D tensors allowed only
    assert x.ndim in [3, 4], "Only 3D and 4D Tensors allowed!"

    # 4D Tensor
    if x.ndim == 4:
        res = []
        for i in range(x.shape[0]):
            img = array2img(x[i])
            img = img.resize((size, size))
            img = np.asarray(img, dtype='float32')
            img = np.expand_dims(img, axis=0)
            img /= 255.0
            res.append(img)
        res = np.concatenate(res)
        res = np.expand_dims(res, axis=1)
        return res

    # 3D Tensor
    img = array2img(x)
    img = img.resize((size, size))
    res = np.asarray(img, dtype='float32')
    res = np.expand_dims(res, axis=0)
    res /= 255.0
    return res


def img2array(data_path, desired_size=None, expand=False, view=False):
    """
    Util function for loading RGB image into a numpy array.

    Returns array of shape (1, H, W, C).
    """
    img = Image.open(data_path)
    img = img.convert('RGB')
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()
    x = np.asarray(img, dtype='float32')
    if expand:
        x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x


def array2img(x):
    """
    Util function for converting anumpy array to a PIL img.

    Returns PIL RGB img.
    """
    x = np.asarray(x)
    x = x + max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    return Image.fromarray(x.astype('uint8'), 'RGB')
