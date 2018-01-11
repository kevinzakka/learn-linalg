## Matrix-Vector Multiplication

If b = Ax, then b is a linear combination of the columns of A.

Let x be an n-dimensional column vector and A be an mxn matrix. Then the matrix-vector product b = Ax is the m dimensional column vector defined as

$$
b_i = sum(j=1 to n) a_ij x_j for i = 1 to m.
$$

Basically, we're mixing the n columns of A (each of dimension m) into a single column vector b of dimension m.

[b] = [a1|a2|...|an][x1]  = x1[a1] + x2[a2] + ... + xn[an]
                    [x2]
                    [..]
                    [xn]

Usually when writing Ax = b, we think of A as acting on x to produce b. In contrast, we can also think about it as x acting on A to produce b.

## Least Squares and When To Use What

Solution of Ax = b is divided into 2 cases:

- A is square
- A is rectangular (i.e. tall, overdetermined, etc.)

When A is rectangular, we can relax our equality and try to find the vector x that minimizes the squared euclidean distance ||Ax-b||. In that case, we're solving least squares.

In any case, we can multiply A on the left by A.T and we obtain a square matrix to which we can apply the decompositions we learned to obtain the solution.

We can solve Ax = b using:

- Normal Equation
- QR decomposition (g-s, householder)
- LU decomposition (pivoting)
- Cholesky decomposition
- SVD

Why is LU used rather than QR to solves square systems of equations?

LU: 2/3 m^3
QR: 4/3 m^3

The factor of 2 is a reason. Also, historically, LU and the process of elimination have been known for centuries whilst QR is relatively recent. There is not enough of a compelling advantage for QR to supplant GE (according to Bau and David).

When A is nxn, i.e. it is a square matrix, then we have a unique solution to the linear system of equations. In that case, we can use LU,

- svd is used because it offers the highest accuracy when A is extremely ill-conditioned
- but in your typical case, since its slow, ur better off with QR using householder

So if writing a least squares solver, you can either:

- solve using SVD, will be slowest but will have best accuracy.
- apply QR decomposition on A directly. Will be faster than SVD and have good accuracy.
- apply LU or Cholesky on A^TA. Cholesky will give the worst accuracy but will be fastest (cond(A^TA) = cond(A^2)). LU with pivoting will be slower than QR but have good accuracy.

Thus LU and QR will do an excellent job and will be faster than SVD which does the best job.


## Gaussian Elimination

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

### Pivoting Strategies

- choosing the pivot is important since dividing by it can lead to instability
    - ideally, we want a big pivot to lead to large numbers
    - if pivot is very small, this can blow up once we scale by the inverse of the pivot
    - we can solve this by doing 2 things before picking a pivot:
        - partial pivoting: look through the current column and permutes rows of the matrix 
          so that the largest absolute value appears on the diagonal.
        - full pivoting: iterate over the entire matrix and permute both rows and columns such
          that diagonal has largest possible values. (way more expensive)

For columns instead of rows, postmultiply A by the permutation matrix `np.dot(A, P)`.

## LU Decomposition

In many engineering applications, when you solve Ax=b, the matrix A remains unchanged, while the right hand side vector b keeps changing (a typical example is when one is solving partial differential equations).

The key idea behind solving using the LU factorization is to decouple the factorization phase (usually computationally expensive) from the actual solving phase. The factorization phase only needs the matrix A, while the actual solving phase makes use of the factored form of A and the right hand side to solve the linear system. Hence, once we have the factorization, we can make use of the factored form of A to solve for different right hand sides at a relatively moderate computational cost.

Given $A = LU$, we have $LUx = b$ and define $y = Ux$. To solve this system, we use a (now familiar) 2-step solution:

- Solve the lower triangular system $Ly = b$ for y using forward substitution.
- Solve the upper triangular system $Ux = y$ for x using back substituion.

This can easily be parallelized for multiple right-hand side b's. We perform the factorization A = LU once, then

- Solve $LY = B$ by parallel forward substitutions.
- Solve $UX = Y$ by parallel back substitutions. 

The cost of factorizing the matrix A into $LU$ is $O(n^3)$. Once you have this factorization, the cost of solving i.e. the cost of solving $LUx=b$ is just $O(n^2)$, since the cost of solving a triangular system scales as $O(n^2)$.

Hence, if you have `r` right hand side vectors $[b_1, b_2, ..., b_r]$, once you have the $LU$ factorization of the matrix A, the total cost to solve is $O(n^3 + rn^2)$ instead of $O(rn^3)$.

This begs the question, how do we perform the factorization step?

TBC

### More into GE

Gaussian elimination transforms a full linear system into an upper-triangular one by applying simple linear transformations on the left. We confine oursevles to square matrices since the algorithm is rarely applied to rectangular matrices in practice.

The idea of GE is to transform the matrix A into an mxm upper-triangular matrix U by introducing zeros below the diagonal, first in column 1, then in column 2, and so on. This is done by subtracting multiples of each row from subsequent rows. This elimination process is equivalent to multiplying A by a sequence of lower-triangular matrices L_k on the left.

Ignoring the permutation operation we learned earlier, recall that if we want to add c times the row k to row l, then this consists in multiplying A on the left by L_k where

L_k = I + cxe_lxe_k.T

Note that since l > k, L_k is always a unit lower-triangular matrix. For example, given a 3x3 matrix A, and say we want to eliminate rows 2 and 3, then our matrix L1 would look like this

[1,  0,  0]
[x,  1,  0]
[x,  0,  1]

and then we want to eliminate row 2 from row 3 then L2 would look like this

[1,  0,  0]
[0,  1,  0]
[0,  x,  1]

By unit, we mean that the elements on the diagonal of L_k are equal to 1. I'll be foregoing the proof here, but we can show that the inverse of L_k is just L_k with the subdiagonal elements negated and finally that L, the product of all L_k's is itself a unit lower-triangular matrix with the subdiagonal entries collected in the appropriate places.

- show that L_k is unit lower-triangular
- show that the inverse of L_k is L_k with the subdiagonal entries negated
- show that the product of unit lower-triangular is unit lower-triangular
- thus L which is the product of all these L_k's is a unit lower-triangular matrix 
  with the subdiagonal entries of all the seperate L_k's in the correct position.

### Determinant of a Matrix

We can compute the determinant of a matrix efficiently by using the LU factorization. Given the LUP decomposition of a matrix

A = P^-1 L U, and the fact that

- the determinant of a triangular matrix is the product of the diagonals.
- the diagonal elements of L are all 1, hence we don't need to compute it's determinant.
- the determinant of P

## Cholesky Factorization

Our discussion of Gaussian Elimination and the LU factorization led to a generic method for solving linear systems of equations. While this strategy always works, sometimes we can gain speed or numerical advantages by examining the particular system we are solving. In particular, we'll see how Hermitian positive definite matrices (and where they arise) can be decomposed into triangular factors twice as quickly as general matrices.

The standard algorithm for this, Cholesky factorization, is a variant of Gaussian elimination that operates on both the left and the right of the matrix at once, preserving and exploiting symmetry.

### Why Cholesky Factorization?

While symmetric, positive definite matrices are rather special, they occur quite frequently in some applications, so their special factorization, called Cholesky decomposition, is good to know about. When you can use it, Cholesky decomposition is about a factor of two faster than alternative methods for solving linear equations.

### Hermitian Positive Definite Matrices

A real matrix A is symmetric if it has the same entries below and above the diagonal. In other words, a_ij = a_ji for all i,j (i != j) or A.T = A. Such a matrix satisfies x.T . A . y = y.T . A . x for all vectors x, y.

More generally, when working with complex matrices, the analoguous of symmetric matrix is hermitian. A hermitian matrix has entries above the diagonal that are the complex conjugate of the entries below. This means that it is equal to its conjugate transpose or A = conj(A.T) or A = A.H where H is the conjugate transpose operation.

Since the diagonal elements stay the same when taking the transpose, it follows necessarily that the diagonal elements of a hermitian matrix must be real, i.e. they are equal to their conjugates.

A hermitian matric A satisfies conj(x).A.y = conj(conj(y).A.x). This means in particular that for any complex x, conj(x).A.x is real. If in addition, conj(x).A.x > 0 for all x != 0, then A is said to be hermitian positive definite.

### Symmetric Gaussian Elimination

Instead of seeking arbitrary lower and upper triangular factors L and U, Cholesky decomposition constructs a lower triangular matrix L whose transpose L.T can itself serve as the upper triangular part (remember that the transpose of a lower triangular matrix is upper triangular).

A = LL^T

This factorization is sometimes referred to as "taking the square root" of the matrix A.

### My attempt

- v 0.1:

In this first attempt, I'll be restricting myself to real matrices, hence symmetric ones.

I'll also implement an inefficient way of computing the Cholesky factorization wherein I apply an elimination matrix on the left than on the right.

- v 0.2:

I implemented the book version. It takes advantage of the fact that A is symmetric, so we just need to work on the super-diagonal part. i.e., after taking the upper diagonal part of A, we act, at each iteration, like we've already left multiplied by the elimination matrix. So all we need to do is scale by the inverse of the square root of the pivot and eliminate the rows underneath. Rinse repeat for all rows.

- v 0.3:

In progress. I need to apply the Cholesky–Crout variant which contains explicit formulas for each entry in the triangular matrix.

### Applications of Cholesky Decomposition

- solving Ax = b using normal equation A^TAx = A^Tb: Not really used in practice as it performs poorly with ill-conditioned A's.
- Monte Carlo simulation: simulates a system with correlated variables. Cholesky decomposition is applied to the correlation matrix, providing a lower triangular matrix L, which when applied to a vector of uncorrelated samples, u, produces the covariance vector of the system.
- non-linear optimization: update the Cholesky decomposition of an approximation of the Hessian matrix.
- matrix inversion

### Linear Systems in the Wild

- aligning 2 images
- unblur using deconvolution


#### Image Alignment

Imagine 2 photographs of the same scene. In the image alignment task, we mark a number of points x (x1, x2) and y (y1, y2) such that x in image 1 corresponds to y in image 2. Since we are bound to make mistakes, it's better to oversample the number of necessary pairs (x, y). 

A reasonable assumption is that there exists some A and a translation vector b such that y = Ax + b. Essentially, an affine transformation has been applied on image 1 to produce image 2.

The unknowns are thus A and b. To solve for them, we can minimize the square of [(Ax + b) - y]. Since we have oversampled our number of points, A will be overdetermined.

Applications include:

- align images that were taken at different times or with different sensors
- align touch points for multi-touch gestures or calibration
- correct images for lens distortion
- correct effects of camera orientation

## QR Decomposition

We have learned about least squares Ax approx b and how a solution x must satisfy the normal equations (A.TA)x=A.Tb. In particular, we have seen how Cholesky decomposition utilizes the special structure of A.TA to quickly solve for x.

One large problem limiting the use of Cholesky is the condition number. It is the square of that of A. Thus, while generic linear strategies might work on A.TA when the least-squares problem is “easy,” when the columns of A are nearly linearly dependent these strategies are likely to generate considerable error since they do not deal with A directly.

### Orthogonality

The least-squares problem is difficult when the columns of A are very similar, i.e. nearly linearly dependent. So when is LS most straightforward?

Well, the easiest linear system is Ix = b, i.e. x=b. This is very unlikely but it could happen that A.T A = I. Let's call such a matrix Q. Such a matrix Q has unit length columns that are orthogonal to one another. They form an orthonormal basis for the column space of Q.

We motivated our discussion by asking when we can expect Q.TQ = I. Now it is easy to see that this occurs when the columns of Q are orthonormal. Furthermore, Q.inv = Q.T. Thus solving Qx = b is as easy as multiplying both sides by the transpose of Q.

Orthonormality also has a strong geometric interpretation. In fact, we can regard orthogonal vectors a and b as being perpendicular. So an orthonormal set of vectors simply is a set od unit-length perpendicular vectors in R^n. 

- If Q is orthogonal, then its action does not affect the length of vectors:

||Qx||^2 = (Qx).T(Qx) = x.TQ.TQx = x.Tx = x.x = ||x||^2

- If Q is orthogonal, Q cannot affect the angle between 2 vectors:

(Qx).(Qy) = (Qx).T (Qy) = x.TQ.TQy = x.Ty = x.y

From this standpoint, if Q is orthogonal, then Q represents an isometry of R^n, that is, it preserves lengths and angles. It can rotate or reflect vectors, but it cannot scale or shear them.

### Strategy for Non-Orthogonal Matrices
















