## linalg

Currently reinforcing my linear algebra and numerical analysis by reimplementing basic, fundamental algorithms in Python. My implementations are tested against `numpy` and `scipy` equivalents. Inspired by [Alex Nichol's Go repository](https://github.com/unixpickle/num-analysis).

## Contents

- [kahan](https://github.com/kevinzakka/linalg/tree/master/kahan): kahan summation for adding finite precision floating point numbers.
- [gelim](https://github.com/kevinzakka/linalg/tree/master/gelim): gaussian elimination with naive, partial and full pivoting for solving `Ax = b`.
- [ludecomp](https://github.com/kevinzakka/linalg/tree/master/ludecomp): `LU`, `PLU` and `PLUQ` decomposition for solving `Ax = b`.
- [determinant](https://github.com/kevinzakka/linalg/blob/master/ludecomp/determinant.py): compute the determinant (or log det) of a square matrix A using PLU factorization.
- [inverse](https://github.com/kevinzakka/linalg/tree/master/inverse): compute the inverse of a square matrix A using PLU factorization.
- [cholesky](https://github.com/kevinzakka/linalg/tree/master/cholesky): cholesky decomposition for symmetric positive definite matrices A.
- [benchmarks](https://github.com/kevinzakka/linalg/tree/master/benchmarks): speed comparisons of different decompositions for solving `Ax = b`.
- [imagealign](https://github.com/kevinzakka/linalg/tree/master/imagealign): align a crooked image using least squares.
- [qrdecomp](https://github.com/kevinzakka/linalg/tree/master/qrdecomp): QR decomposition of any matrix A using `gram-schmidt` or `householder`.
- [lstsq](https://github.com/kevinzakka/linalg/tree/master/lstsq): solve least squares using QR decomposition.

## Todo

- `eigen`: implement various algorithms for finding eigepairs of matrices.
- `svd`: singular value decomposition
- `pinv`: find the Moore-Penrose inverse of a matrix
- `solve`: implement a general solver for a well-determined linear system `Ax=b`
- `multi_dot`: automatically select the fastest evaluation order of a series of dot products using dynamic programming and carry it out.
- `wtyw`: a pdf file explaining *when to use what* in terms of factorizations for solving `Ax=b`.
- Refactor useful functions such as:
    - `norm`: compute different matrix or vector norms.
    - `trace`: compute the trace of an array.
    - `det`: compute the determinant of an array.
    - `cond`: compute the condition number of an array.

## Resources

- [Stanford CS 205A Notes](https://graphics.stanford.edu/courses/cs205a-13-fall/assets/notes/cs205a_notes.pdf)
- [Numerical Linear Algebra](https://www.amazon.com/Numerical-Linear-Algebra-Lloyd-Trefethen/dp/0898713617)
- [Matrix Computations](https://www.amazon.com/Computations-Hopkins-Studies-Mathematical-Sciences/dp/1421407949/)
- [Numerical Recipes: The Art of Scientific Computing](http://numerical.recipes/)