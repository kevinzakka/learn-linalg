import numpy as np

from utils import multi_dot
from row_ops import permute, scale, eliminate


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

    def __init__(self, A, b, pivoting=None):
        self.A = A
        self.b = b
        self.ops = []

        # ensure correct pivoting provided
        error_msg = "[!] Invalid pivoting option."
        allowed = [None, 'partial', 'full']
        assert (pivoting in allowed), error_msg
        self.pivoting = pivoting

    def forward_sub(self):
        num_rows, num_cols = self.A.shape

        for i in range(num_rows):

            # strategy for the first row
            if i == 0:
                # if left-most element is nonzero choose it
                if self.A[i, 0] != 0:
                    self.pivot = self.A[i, 0]
                # else just switch with any row underneath that has nonzero
                else:
                    # find the index of row to switch with
                    for k in range(i+1, num_rows):
                        if self.A[k, 0] != 0:
                            break

                    # use permutation matrix to switch
                    P = permute(num_rows, [(i, k)])
                    self.A = np.dot(P, self.A)
                    self.ops.append(P)

                    # now pivot is in upper left corner
                    self.pivot = self.A[i, 0]
            else:
                # strategy for remaining rows
                self.pivot = None

                if self.pivoting is None:
                    # look through columns of row to find first nonzero element
                    for j in range(1, num_cols):
                        if self.A[i, j] != 0:
                            self.pivot = self.A[i, j]
                            break

                elif self.pivoting == 'partial':
                    # as before, look through column for first nonzero element
                    for j in range(1, num_cols):
                        if self.A[i, j] != 0:
                            pivot_val = np.abs(self.A[i, j])
                            pivot_row = i
                            pivot_col = j
                            break

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
                            self.ops.append(P)

                    self.pivot = self.A[i, pivot_col]

                # full pivoting
                else:
                    pass

            # scale the row containing pivot to make 1
            scale_factor = 1. / self.pivot
            S = scale(num_rows, [(i, scale_factor)])
            self.A = np.dot(S, self.A)
            self.ops.append(S)

            if i != num_rows - 1:
                # eliminate all elements in column underneath pivot
                for k in range(i+1, num_rows):
                    # if element is 0, then done
                    if self.A[k, i] == 0:
                        continue
                    # else eliminate the current row
                    else:
                        # compute scaling factor
                        scale_factor = - (self.A[k, i])
                        # scale row i by this factor and add it to row k
                        E = eliminate(num_rows, i, scale_factor, k)
                        self.A = np.dot(E, self.A)
                        self.ops.append(E)

    def back_sub(self):
        num_rows, num_cols = self.A.shape

        for i in range(num_rows-1, 0, -1):

            # eliminate all elements in column above
            for k in range(i-1, -1, -1):
                # if element is 0, then done
                if self.A[k, i] == 0:
                    continue
                # else eliminiate the current row
                else:
                    # compute scaling factor
                    scale_factor = - (self.A[k, i])
                    # scale row i by this factor and add it to row k
                    E = eliminate(num_rows, i, scale_factor, k)
                    self.A = np.dot(E, self.A)
                    self.ops.append(E)

        # reverse the order of operations
        self.ops = self.ops[::-1]

    def solve(self):
        # perform forward and backward sub
        self.forward_sub()
        self.back_sub()

        # solve for x
        M = multi_dot(self.ops)
        x = np.dot(M, self.b)

        return [x, self.ops]
