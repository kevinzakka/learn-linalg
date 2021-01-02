import numpy as np


class KahanSum:
  """Precise summation of finite-precision floating point numbers [1].

  Reduces numerical error by storing a running compensation term that captures
  lost low-order bits.

  References:
    [1]: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
  """
  def __init__(self):
    """Constructor."""
    self.reset()

  def reset(self):
    """Clears the internal state."""
    # Create one variable for keeping track of the sum and one for storing the
    # moving compensation term.
    self._sum = 0
    self._compensation = 0

  def add(self, x):
    """Adds the float x to the summation term."""
    x += self._compensation
    sum = self._sum + x
    self.compensation = x - (sum - self._sum)
    self._sum = sum

  def result(self):
    return self._sum


def kahan_sum(x, axis=None, keepdims=False):
  """Kahan summation for 1 and 2D arrays.

  Args:
    x: A 1D or 2D array-like object.
    axis: The axis which will be collapsed to perform the summation.
    keepdims: A bool specifying whether to keep the collapsed axis.

  Returns:
    The kahan summation of x.
  """
  # Ensure the array-like object is at most 2D.
  x = np.asarray(x)
  error_msg = "[!] Only 1D and 2D arrays are currently supported."
  assert (x.ndim <= 2), error_msg

  # Sanity check axis args.
  error_msg = "[!] Axis value can only be None, 0 or 1."
  assert (axis in [None, 0, 1]), error_msg

  # Instantiate summation object.
  summation = KahanSum()

  # 1D case.
  if x.ndim == 1:
    for i in range(len(x)):
      summation.add(x[i])
    return summation.result()

  # 2D case.
  num_rows, num_cols = x.shape

  if axis is None:
    for i in range(num_rows):
      for j in range(num_cols):
        summation.add(x[i, j])
    result = summation.result()

  elif axis == 0:
    # This list will hold num_cols sums.
    sums = []
    for i in range(num_cols):
      summation.reset()
      for j in range(num_rows):
        summation.add(x[j, i])
      sums.append(summation.result())
    result = np.asarray(sums)
    if keepdims:
      result = result.reshape([1, num_cols])

  else:
    # This list will hold num_rows sums.
    sums = []
    for i in range(num_rows):
      summation.reset()
      for j in range(num_cols):
        summation.add(x[i, j])
      sums.append(summation.result())
    result = np.asarray(sums)
    if keepdims:
      result = result.reshape([num_rows, 1])

  return result
