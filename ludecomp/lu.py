import numpy as np

from utils import unit_diag, lower_diag, upper_diag
from linalg.gelim.row_ops import permute


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
    - P: a permutation matrix of shape (N, N) if
      partial or full pivoting.
    - Q: a permutation matrix of shape (N, N) if
      full pivoting.
    """

    def __init__(self, pivoting=None):
        error_msg = "[!] Invalid pivoting option."
        allowed = [None, 'partial', 'full']
        assert (pivoting in allowed), error_msg
        self.pivoting = pivoting

    def decompose(self):

        m = len(self.A)

        for i in range(m-1):
            # skip iteration if nothing to be done
            done = True
            if self.A[i, i] == 1:
                for k in range(i+1, m):
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
                    for k in range(i+1, m):
                        if self.A[k, i] != 0:
                            break
                    if k == m - 1:
                        raise Exception("Such a system is inconsistent!")
                        return

                    # use permutation matrix to switch
                    P = permute(m, [(i, k)])
                    self.A = np.dot(P, self.A)
                    self.P = np.dot(P, self.P)

                self.pivot = self.A[i, i]

            elif self.pivoting == 'partial':
                pivot_val = np.abs(self.A[i, i])
                pivot_row = i
                pivot_col = i

                # last row does not need partial pivoting
                if i != m - 1:

                    # look underneath and find bigger pivot if it exists
                    for k in range(i+1, m):
                        if np.abs(self.A[k, pivot_col]) > pivot_val:
                            pivot_val = np.abs(self.A[k, pivot_col])
                            pivot_row = k

                    # switch current row with row containing max
                    if pivot_row != i:
                        P = permute(m, [(i, pivot_row)])
                        self.A = np.dot(P, self.A)
                        self.P = np.dot(P, self.P)

                self.pivot = self.A[i, pivot_col]

            # full pivoting
            else:
                pivot_val_col = self.A[i, i]
                pivot_val_row = self.A[i, i]
                pivot_row = i
                pivot_col = i

                # last row does not need full pivoting
                if i != m - 1:

                    # look through right of row for largest pivot
                    for j in range(i+1, m):
                        if np.abs(self.A[i, j]) > pivot_val_col:
                            pivot_val_col = np.abs(self.A[i, j])
                            pivot_col = j

                    # look through bottom of col for largest pivot
                    for k in range(i+1, m):
                        if np.abs(self.A[k, i]) > pivot_val_row:
                            pivot_val_row = np.abs(self.A[k, i])
                            pivot_row = k

                    # switch current row with row containing max
                    if pivot_row != i or pivot_col != i:
                        # choose the largest of both
                        if pivot_val_col < pivot_val_row:
                            P = permute(m, [(i, pivot_row)])
                            self.A = np.dot(P, self.A)
                            self.P = np.dot(P, self.P)
                        else:
                            Q = permute(m, [(i, pivot_col)])
                            self.A = np.dot(self.A, Q)
                            self.Q = np.dot(self.Q, Q)

                self.pivot = self.A[i, i]

            for j in range(i+1, m):
                scale_factor = (self.A[j, i] / self.pivot)
                self.A[j, i] = scale_factor
                for k in range(i+1, m):
                    self.A[j, k] -= scale_factor*self.A[i, k]

    def __call__(self, A):
        self.A = A
        self.P = np.eye(len(A))
        self.Q = np.eye(len(A))

        self.decompose()

        P = self.P
        L = unit_diag(lower_diag(self.A))
        U = upper_diag(self.A, diag=True)
        Q = self.Q

        if self.pivoting is None:
            return (L, U)
        elif self.pivoting == "partial":
            return (P, L, U)
        return (P, L, U, Q)
