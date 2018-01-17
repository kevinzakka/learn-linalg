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
- `misc`: refactor useful functions such as:
    - `norm`: different matrix or vector norms.
    - `trace`: return the trace of an array.
    - `det`: return the determinant of an array.
    - `cond`: return the condition number of an array.

Finally, if I get the chance, I'd like to reimplement basic operations such as the dot product, matrix multiplication, saxpy and calculate the number of FLOPS to get a feel for how they can be made more efficient.

## Resources

- [Stanford CS 205A Notes](https://graphics.stanford.edu/courses/cs205a-13-fall/assets/notes/cs205a_notes.pdf)
- [Numerical Linear Algebra](https://www.amazon.com/Numerical-Linear-Algebra-Lloyd-Trefethen/dp/0898713617)
- [Numerical Recipes: The Art of Scientific Computing](http://numerical.recipes/)