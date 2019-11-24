import numpy as np


class KahanSum:
  """Precise summation of finite-precision floating point numbers [1].

  Reduces numerical error by storing a running compensation term
  that captures lost low-order bits.

  References:
    [1]: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
  """
  def __init__(self):
    """Initializes the KahanSum object.

    Creates 2 variables: one for keeping track of the sum
    and one for storing the moving compensation term.
    """
    self.sum = 0.
    self.c = 0.

  def add(self, x):
    """Adds the float x to the sum term.
    """
    # add back compensation
    x += self.c

    # add to sum
    sum = self.sum + x

    # update compensation
    self.c = x - (sum - self.sum)

    # update sum
    self.sum = sum

    return self.sum

  def cur_sum(self):
    return self.sum


def kahan_sum(x, axis=None, keepdims=False):
  """Kahan summation of a list of floating point numbers.

  Args:
    x: a numpy ndarray of (M, N)
    axis: the axis which will be collapsed to perform the summation.
      None: sums all the elements in x.
      0: sums the elements in each column and returns a sum
        of shape (N,).
      1: sums the elements in each row and returns a sum
        of shape (M, ).
    keepdims: bool specifying whether to keep the collapsed axis
      with a value of 1.

  Returns:
    sum: kahan summation of x.
  """
  x = np.asarray(x)

  error_msg = "[!] Only 1D and 2D arrays are currently supported."
  assert (x.ndim <= 2), error_msg

  # 1D case
  if x.ndim == 1:
    N = len(x)

    # instantiate a single KahanSum object
    summation = KahanSum()

    # loop over rows and columns of x:
    for i in range(N):
      summation.add(x[i])

    return summation.cur_sum()

  # 2D case
  else:
    num_rows, num_cols = x.shape

    if axis is None:
      # instantiate a single KahanSum object
      summation = KahanSum()

      # loop over rows and columns of x:
      for i in range(num_rows):
        for j in range(num_cols):
          summation.add(x[i, j])

      return summation.cur_sum()

    elif axis == 0:
      # this list will hold num_cols sums
      sums = []

      # loop over columns of x
      for i in range(num_cols):
        # instantiate a KahanSum object
        summation = KahanSum()

        # loop over rows of x:
        for j in range(num_rows):
          summation.add(x[j, i])

        sums.append(summation.cur_sum())

      summation = np.asarray(sums)
      if keepdims:
        summation = summation.reshape([1, num_cols])
      return summation

    elif axis == 1:
      # this list will hold num_rows sums
      sums = []

      # loop over rows of x
      for i in range(num_rows):
        # instantiate a KahanSum object
        summation = KahanSum()

        # loop over columns of x:
        for j in range(num_cols):
          summation.add(x[i, j])

        sums.append(summation.cur_sum())

      summation = np.asarray(sums)
      if keepdims:
        summation = summation.reshape([num_rows, 1])
      return summation

    else:
      raise ValueError("Axis value can only be None, 0 or 1.")