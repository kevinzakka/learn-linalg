import numpy as np

from sum import KahanSum
from row_ops import permute
from utils import unit_diag, lower_diag, upper_diag


class LU(object):
    """
    Performs the LU decomposition of a matrix A. This is useful
    for solving linear systems of equations with multiple right
    hand side vectors b.

    LU decomposition decouples the factorization phase from the actual
    solving phase such that the factorization can be reused to efficiently
    solve for each b.

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
    - b: a column vector of shape (N,). Can also be multiple column
      vectors in the case where one wishes to solve many right-hand
      sides associated with the same A. In that case, b is of shape
      (N, K).

    Returns
    -------
    - L: a lower triangular matrix of shape (N, N).
    - U: an upper triangular matrix of shape (N, N).
    - P: a permutation matrix of shape (N, N) if
      partial or full pivoting is used.
    - Q: a permutation matrix of shape (N, N) if
      full pivoting is used.
    """

    def __init__(self, A, pivoting=None):
        self.backup = np.array(A)
        error_msg = "[!] Invalid pivoting option."
        allowed = [None, 'partial', 'full']
        assert (pivoting in allowed), error_msg
        self.pivoting = pivoting

    def decompose(self, ret=True, det=False):

        N = len(self.backup)
        self.A = np.array(self.backup)
        self.P = np.eye(N)
        self.Q = np.eye(N)
        self.num_switches = 0

        for i in range(N-1):
            # skip iteration if nothing to be done
            done = True
            if self.A[i, i] == 1:
                for k in range(i+1, N):
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
                    for k in range(i+1, N):
                        if self.A[k, i] != 0:
                            break
                    if k == N - 1:
                        raise Exception("Such a system is inconsistent!")
                        return

                    # use permutation matrix to switch
                    P = permute(N, [(i, k)])
                    self.A = np.dot(P, self.A)
                    self.P = np.dot(P, self.P)
                    self.num_switches += 1

                self.pivot = self.A[i, i]

            elif self.pivoting == 'partial':
                pivot_val = np.abs(self.A[i, i])
                pivot_row = i
                pivot_col = i

                # last row does not need partial pivoting
                if i != N - 1:

                    # look underneath and find bigger pivot if it exists
                    for k in range(i+1, N):
                        if np.abs(self.A[k, pivot_col]) > pivot_val:
                            pivot_val = np.abs(self.A[k, pivot_col])
                            pivot_row = k

                    # switch current row with row containing max
                    if pivot_row != i:
                        P = permute(N, [(i, pivot_row)])
                        self.A = np.dot(P, self.A)
                        self.P = np.dot(P, self.P)
                        self.num_switches += 1

                self.pivot = self.A[i, pivot_col]

            # full pivoting
            else:
                pivot_val_col = self.A[i, i]
                pivot_val_row = self.A[i, i]
                pivot_row = i
                pivot_col = i

                # last row does not need full pivoting
                if i != N - 1:

                    # look through right of row for largest pivot
                    for j in range(i+1, N):
                        if np.abs(self.A[i, j]) > pivot_val_col:
                            pivot_val_col = np.abs(self.A[i, j])
                            pivot_col = j

                    # look through bottom of col for largest pivot
                    for k in range(i+1, N):
                        if np.abs(self.A[k, i]) > pivot_val_row:
                            pivot_val_row = np.abs(self.A[k, i])
                            pivot_row = k

                    # switch current row with row containing max
                    if pivot_row != i or pivot_col != i:
                        # choose the largest of both
                        if pivot_val_col < pivot_val_row:
                            P = permute(N, [(i, pivot_row)])
                            self.A = np.dot(P, self.A)
                            self.P = np.dot(P, self.P)
                            self.num_switches += 1
                        else:
                            Q = permute(N, [(i, pivot_col)])
                            self.A = np.dot(self.A, Q)
                            self.Q = np.dot(self.Q, Q)
                            self.num_switches += 1

                self.pivot = self.A[i, i]

            for j in range(i+1, N):
                scale_factor = (self.A[j, i] / self.pivot)
                self.A[j, i] = scale_factor
                for k in range(i+1, N):
                    self.A[j, k] -= scale_factor*self.A[i, k]

        self.P = self.P
        self.L = unit_diag(lower_diag(self.A))
        self.U = upper_diag(self.A, diag=True)
        self.Q = self.Q

        if ret:
            if det:
                return (self.num_switches, self.U)
            if self.pivoting is None:
                return (self.L, self.U)
            elif self.pivoting == "partial":
                return (self.P.T, self.L, self.U)
            return (self.P.T, self.L, self.U, self.Q.T)

    def solve(self, b):
        """
        Perform the LU factorization on the matrix A
        and then solve the linear system Ax = b using
        forward and backward substitution.
        """
        self.b = b

        self.decompose(ret=False)
        self._forward()
        self._backward()

        return self.x

    def _forward(self):
        """
        Solves the lower triangular system Ly = b
        for y by forward substitution.

        If partial pivoting is used, solves the system
        Ly = Pb and if full pivoting is used, solves
        the system Ly = PbQ.
        """

        if self.b.ndim > 1:
            num_iters = self.b.shape[1]
            N = self.b.shape[0]
        else:
            num_iters = 1
            N = self.b.shape[0]

        self.y = np.zeros([N, num_iters])

        if self.pivoting is None:
            right_hand = self.b
        elif self.pivoting == "partial":
            right_hand = np.dot(self.P, self.b)
        else:
            right_hand = np.dot(self.P, self.b)
            right_hand = np.dot(right_hand[:, np.newaxis].T, self.Q)
            right_hand = right_hand.squeeze().T

        for k in range(num_iters):
            for i in range(N):
                acc = KahanSum()
                for j in range(i):
                    acc.add(self.L[i, j]*self.y[j, k])
                if self.b.ndim > 1:
                    self.y[i, k] = right_hand[i, k] - acc.cur_sum()
                else:
                    self.y[i, k] = right_hand[i] - acc.cur_sum()

    def _backward(self):
        """
        Solve the upper triangular system Ux = y
        for x by back substitution.
        """

        if self.b.ndim > 1:
            num_iters = self.b.shape[1]
            N = self.b.shape[0]
        else:
            num_iters = 1
            N = self.b.shape[0]

        self.x = np.zeros([N, num_iters])

        for k in range(num_iters):
            for i in range(N-1, -1, -1):
                acc = KahanSum()
                for j in range(N-1, i, -1):
                    acc.add(self.U[i, j]*self.x[j, k])
                self.x[i, k] = (self.y[i, k] - acc.cur_sum()) / (self.U[i, i])

        if self.b.ndim == 1:
            self.x = self.x.squeeze()
