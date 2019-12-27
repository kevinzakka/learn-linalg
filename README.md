## linalg

Currently reinforcing my linear algebra and numerical analysis by reimplementing basic, fundamental algorithms in Python. My implementations are tested against `numpy` and `scipy` equivalents. Inspired by [Alex Nichol's Go repository](https://github.com/unixpickle/num-analysis).

Feel free to read the [notes](https://github.com/kevinzakka/learn-linalg/blob/master/linalg/notes.md) which summarize parts of Justin Solomon's [book](https://people.csail.mit.edu/jsolomon/share/book/numerical_book.pdf) as well as insights into my thought-process.

## Contents

- [kahan](https://github.com/kevinzakka/learn-linalg/tree/master/linalg/kahan): kahan summation for adding finite precision floating point numbers.
- [gelim](https://github.com/kevinzakka/learn-linalg/tree/master/linalg/gelim): gaussian elimination with naive, partial and full pivoting for solving `Ax = b`.
- [ludecomp](https://github.com/kevinzakka/learn-linalg/tree/master/linalg/ludecomp): `LU`, `PLU` and `PLUQ` decomposition for solving `Ax = b`.
- [determinant](https://github.com/kevinzakka/learn-linalg/blob/master/linalg/misc/determinant.py): compute the determinant (or log det) of a square matrix A using PLU factorization.
- [inverse](https://github.com/kevinzakka/learn-linalg/tree/master/linalg/misc/inverse.py): compute the inverse of a square matrix A using PLU factorization.
- [cholesky](https://github.com/kevinzakka/learn-linalg/tree/master/linalg/cholesky): cholesky decomposition for symmetric positive definite matrices A.
- [qrdecomp](https://github.com/kevinzakka/learn-linalg/tree/master/linalg/qrdecomp): QR decomposition of any matrix A using `gram-schmidt` or `householder`.
- [solve](https://github.com/kevinzakka/learn-linalg/tree/master/linalg/solver): solve `Ax=b` using PLU decomposition.
- [lstsq](https://github.com/kevinzakka/learn-linalg/tree/master/linalg/lstsq): solve least squares using QR decomposition.
- [eigen](https://github.com/kevinzakka/learn-linalg/tree/master/linalg/eigen): single and multi eigenvalue finding algorithms.

## Applications

- [imagealign](examples/imagealign/): align a crooked image using least squares.
- [benchmarks](examples/benchmarks/): speed comparisons of different decompositions for solving `Ax = b`.

## Resources

- [Stanford CS 205A Notes](https://graphics.stanford.edu/courses/cs205a-13-fall/assets/notes/cs205a_notes.pdf)
- [Numerical Linear Algebra](https://www.amazon.com/Numerical-Linear-Algebra-Lloyd-Trefethen/dp/0898713617)
- [Numerical Recipes: The Art of Scientific Computing](http://numerical.recipes/)

## Todos

- Implement SVD.
- Implement conjugate gradient.
- Make a deblurify application using conjugate gradient.