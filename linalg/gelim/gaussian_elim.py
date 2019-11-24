import numpy as np

from linalg.gelim.row_ops import permute, scale, eliminate


class GaussElim(object):
  """
  Apply Gaussian Elimination to solve a system of linear
  equations of the form Ax = b.

  Concretely, GaussElim proceeds in 2 steps:

  - forward subsitution: for each row in the matrix A, if the
    row does not consist of zeros, pick the  left-most non-zero
    entry as the pivot. If the row is full of zeros, swap it with
    any non-zero lower row. Apply a scaling operation to so that
    the pivot is equal to 1, then use this row to eliminate all
    other values underneath the same column. Finally, move the
    pivot to the next row and repeat a similar series of operations
    until the matrix A is an upper-tiangular matrix.

  - back substitution: proceed in the reverse order of rows and
    eliminate backwards such that the end result is the indentity
    matrix.

  Note: This needs to be rewritten to remove explicit elimination
  matrices.

  Args
  ----
  - A: a numpy array of shape (N, N).
  - b: a numpy array of shape (N,).
  - pivoting: (string) the pivoting strategy used. If None, performs
    a naive strategy that picks the left-most nonzero entry.
    - 'partial': looks through the current column and permutes
      rows of the matrix so that the largest absolute value
      appears on the diagonal.
    - 'full': iterates over the entire matrix and permutes both
      rows and columns to get the largest possible value on the
      diagonal.

  Returns
  -------
  - x: a numpy array of shape (N,)
  - M: the inverse of the matrix A which corresponds to the dot
    product of all the elementary row operations applied during
    the forward and back substitution in reverse order. It is a
    numpy array of shape (N, N).
  """

  def __init__(self, A, b, pivoting=None, record=False):
    self.A = A
    self.b = b
    self.M = np.eye(A.shape[0])

    error_msg = "[!] Invalid pivoting option."
    allowed = [None, 'partial', 'full']
    assert (pivoting in allowed), error_msg
    self.pivoting = pivoting

  def forward(self):
    num_rows, num_cols = self.A.shape

    for i in range(num_rows):
      # skip iteration if nothing to be done
      done = True
      if self.A[i, i] == 1:
        for k in range(i+1, num_rows):
          if self.A[k, i] != 0:
            done = False
      else:
        done = False

      if done:
        continue

      # determine the pivot based on pivoting strategy
      if self.pivoting is None:
        # switch with any below row if zero
        if self.A[i, i] == 0:
          # find the index of row to switch with
          for k in range(i+1, num_rows):
            if self.A[k, i] != 0:
              break
          if k == num_rows - 1:
            raise Exception("Such a system is inconsistent!")
            return

          # use permutation matrix to switch
          P = permute(num_rows, [(i, k)])
          self.A = np.dot(P, self.A)
          self.M = np.dot(P, self.M)

        self.pivot = self.A[i, i]

      elif self.pivoting == 'partial':
        pivot_val = np.abs(self.A[i, i])
        pivot_row = i
        pivot_col = i

        # last row does not need partial pivoting
        if i != num_rows - 1:

          # look underneath and find bigger pivot if it exists
          for k in range(i+1, num_rows):
            if np.abs(self.A[k, pivot_col]) > pivot_val:
              pivot_val = np.abs(self.A[k, pivot_col])
              pivot_row = k

          # switch current row with row containing max
          if pivot_row != i:
            P = permute(num_rows, [(i, pivot_row)])
            self.A = np.dot(P, self.A)
            self.M = np.dot(P, self.M)

        self.pivot = self.A[i, pivot_col]

      # full pivoting
      else:
        pivot_val_col = self.A[i, i]
        pivot_val_row = self.A[i, i]
        pivot_row = i
        pivot_col = i

        # last row does not need full pivoting
        if i != num_rows - 1:

          # look through right of row for largest pivot
          for j in range(i+1, num_cols):
            if np.abs(self.A[i, j]) > pivot_val_col:
              pivot_val_col = np.abs(self.A[i, j])
              pivot_col = j

          # look through bottom of col for largest pivot
          for k in range(i+1, num_rows):
            if np.abs(self.A[k, i]) > pivot_val_row:
              pivot_val_row = np.abs(self.A[k, i])
              pivot_row = k

          # switch current row with row containing max
          if pivot_row != i or pivot_col != i:
            # choose the largest of both
            if pivot_val_col < pivot_val_row:
              P = permute(num_rows, [(i, pivot_row)])
              self.A = np.dot(P, self.A)
              self.M = np.dot(P, self.M)
            else:
              P = permute(num_rows, [(i, pivot_col)])
              self.A = np.dot(self.A, P)
              self.M = np.dot(self.M, P)

        self.pivot = self.A[i, i]

      # scale the row containing pivot to make 1
      scale_factor = 1. / self.pivot
      S = scale(num_rows, [(i, scale_factor)])
      self.A = np.dot(S, self.A)
      self.M = np.dot(S, self.M)

      l_s = list(np.arange(i+1, num_rows))
      c_s = [-self.A[k, i] for k in range(i+1, num_rows)]
      E = eliminate(num_rows, i, c_s, l_s)
      self.A = np.dot(E, self.A)
      self.M = np.dot(E, self.M)

  def backward(self):
    num_rows, num_cols = self.A.shape

    for i in range(num_rows-1, 0, -1):
      l_s = list(np.arange(i-1, -1, -1))
      c_s = [-self.A[k, i] for k in range(i-1, -1, -1)]
      E = eliminate(num_rows, i, c_s, l_s)
      self.A = np.dot(E, self.A)
      self.M = np.dot(E, self.M)

  def solve(self):
    # perform forward and backward sub
    self.forward()
    self.backward()

    x = np.dot(self.M, self.b)

    return [x, self.M]
