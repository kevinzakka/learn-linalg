## linalg

Currently reinforcing my linear algebra and numerical analysis by reimplementing basic, fundamental algorithms in Python. My implementations are tested against `numpy` and `scipy` equivalents. Inspired by [Alex Nichol's Go repository](https://github.com/unixpickle/num-analysis).

Feel free to read the [notes](linalg/notes.md) which summarize parts of Justin Solomon's [book](https://people.csail.mit.edu/jsolomon/share/book/numerical_book.pdf) as well as insights into my thought-process.

## Contents

- [kahan](linalg/kahan): kahan summation for adding finite precision floating point numbers.
- [gelim](linalg/gelim): gaussian elimination with naive, partial and full pivoting for solving `Ax = b`.
- [ludecomp](linalg/ludecomp): `LU`, `PLU` and `PLUQ` decomposition for solving `Ax = b`.
- [determinant](linalg/misc/determinant.py): compute the determinant (or log det) of a square matrix A using PLU factorization.
- [inverse](linalg/misc/inverse.py): compute the inverse of a square matrix A using PLU factorization.
- [cholesky](linalg/cholesky): cholesky decomposition for symmetric positive definite matrices A.
- [qrdecomp](linalg/qrdecomp): `QR` decomposition of any matrix A using gram-schmidt or householder.
- [solve](linalg/solver): solve `Ax=b` using PLU decomposition.
- [lstsq](linalg/lstsq): solve least squares using QR decomposition.
- [eigen](linalg/eigen): single and multi eigenvalue finding algorithms, hessenberg factorization and the qr algorithm.
- [svd](linalg/svd): singular value decomposition `SVD` of any matrix A.
- [optim](linalg/optim/): iterative linear solvers such as gradient descent and conjugate gradients.

## Applications

- [imagealign](examples/imagealign/): align a crooked image using least squares.
- [deblur](examples/deblur/): deblur an image by inverting it using conjugate gradients.
- [benchmarks](examples/benchmarks/): speed comparisons of different decompositions for solving `Ax = b`.

## Resources

- [Stanford CS 205A Notes](https://graphics.stanford.edu/courses/cs205a-13-fall/assets/notes/cs205a_notes.pdf)
- [Numerical Linear Algebra](https://www.amazon.com/Numerical-Linear-Algebra-Lloyd-Trefethen/dp/0898713617)
- [Numerical Recipes: The Art of Scientific Computing](http://numerical.recipes/)

## Todos

- [ ] Make QR decomposition more efficient for Hessenberg matrices.
- [ ] Implement QR decomposition with Givens rotations.
