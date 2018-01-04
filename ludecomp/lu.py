import numpy as np

from gelim.row_ops import permute, scale, eliminate


class LU(object):
    """
    Performs the LU decomposition of a matrix A. This is useful
    for solving linear systems of equations with multiple right
    hand side vectors b.

    LU decomposition decouples the factorization phase from the actual
    solving phase such that the factorization can be reused to efficiently
    solve for each b.

    Concretely, LU decomposition consists in the forward substitution phase
    of Gaussian Elimination with an added step of recording an extra value
    in the places where the zeros are produced. The matrix A is edited in 
    place such that the end result is L and U both being stored in the 
    matrix A.

    Args
    ----
    - A: a numpy array of shape (N, N).
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
    - L: a lower triangular matrix of shape (N, N).
    - U: an upper triangular matrix of shape (N, N).
    """

    def __init__(self, pivoting=None):
        error_msg = "[!] Invalid pivoting option."
        allowed = [None, 'partial', 'full']
        assert (pivoting in allowed), error_msg
        self.pivoting = pivoting

    def decompose(self):

        num_rows, num_cols = self.A.shape

        for i in range(num_rows-1):
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

            # only if left-most element is zero switch with rows below
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

            # now we determine the pivot based on pivoting strategy
            if self.pivoting is None:
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
                        else:
                            P = permute(num_rows, [(i, pivot_col)])
                            self.A = np.dot(self.A, P)

                self.pivot = self.A[i, i]

            for k in range(i+1, num_rows):
                scale_factor = -(self.A[k, i]/self.pivot)
                self.L[k, i] = - scale_factor
                E = eliminate(num_rows, i, scale_factor, k)
                self.A = np.dot(E, self.A)

        self.U = self.A

    def __call__(self, A):
        self.A = A
        self.L = np.eye(A.shape[0])
        self.decompose()
        return self.L, self.U
