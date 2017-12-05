Ax = b

- A of shape (M, N)
- x of shape (N,)
- b of shape(M,)

3 types of solutions
- no solution (non-compatibility)
- a unique solution
- infinitely many solutions (underdetermined)

- shape of A influences the type of solution
    - if A is wide (n > m), then it cannot admit a unique solution.
    - if A is tall (m > n), then it admits unsolvable solutions.

- we consider square matrices A (N, N)
- we consider A is nonsingular, i.e. invertible

- Solving linear systems involves applying only 3 operations:
    - permuatation (we use a permutation matrix for this: P x A)
        - e_k is the standard basis function
        - e_k.T x A picks out the k-th row of A
        - we stack a bunch of e_k's vertically to obtain permutation matrix P
        - ex: if we want to permute a matrix A that is 3x3 such that we want
              row 2, then row 3 then row 1.
        - we would construct the following matrix P: P = [0 1 0
                                                          0 0 1
                                                          1 0 0]
    - row scaling (we use a scaling matrix S: S x A)
        - ex: suppose we want to scale the thid row of A by 0.5.
        - we could construct S = diag(1, 1, 0.5)
    - adding the scale of a row to another: (we use an elimination matrix M: M x A)
        - e_k.T x A picks out the k'th row from A
        - e_l x e_k.T x A yields a matrix which is zero everywhere excet the l'th row
          which is equal to the k'th row of A.
        - ex: suppose we wish to isolate the third row of A and move it to row 2
        - We would do this with the following operation: e_2 x e_3.T x A
        - ex: suppose we want to add c times the row k to row l
        - A + (c x e_l x e_k.T x A) = (I + c x e_l x e_k.T) x A

- more generally, to solve linear systems, we will use the Gaussian Elimination algorithm:
    - Forward Substitution
        - start with row 1 pivot:
            - if zero, switch with any row that has nonzero first element
            - scale to have a pivot of 1
            - apply elimination using row with this pivot to make values underneat this pivot 0
        - go to row 2 and pick 2nd element as pivot:
            - rinse and repeat
        - the end result should be an upper-triangular matrix
    - Back Substitution
        - proceed in reverse order of rows and eliminate backward
        - end result should be the identity matrix

- each GE op takes O(n) since requires iterating over n elements (at most 2n)
- for each pivot, we have forward and backward so O(n^2)
- since n pivots, we have O(n^3)

- choosing the pivot is important since dividing by it can lead to instability
    - ideally, we want a big pivot to lead to large numbers
    - if pivot is very small, this can blow up once we scale by the inverse of the pivot
    - we can solve this by doing 2 things before picking a pivot:
        - partial pivoting: look through the current column and permutes rows of the matrix 
          so that the largest absolute value appears on the diagonal.
        - full pivoting: iterate over the entire matrix and permute both rows and columns such
          that diagonal has largest possible values. (way more expensive)

For columns instead of rows, postmultiply A by the permutation matrix `np.dot(A, P)`.