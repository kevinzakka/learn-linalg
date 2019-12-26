import numpy as np

from linalg.kahan.sum import KahanSum
from functools import reduce


def projection(b, a):
  """Compute the projection of b onto a.

  The projection of b onto a is the orthogonal
  projection of b onto a straight line parallel to a.
  The projection is parallel to a, i.e. it is the product
  of a constant called the scalar projection with a unit
  vector in the direction of a:

  `proj(b, a) = c * a = [(a.T b) /(a.T a)] * a`

  Args:
    b: a numpy array of shape (N, 1).
    a: a numpy array of shape (N, 1).

  Returns:
    proj: a numpy array of shape (N, 1).
  """
  norm_sq = a @ a
  if np.isclose(norm_sq, 1.):
    return (a.T @ b) * a
  c = (a @ b) / norm_sq
  return c * a


def reflection(x, u, apply=False):
  """Reflect the vector x over the vector u.

  A reflection is a linear transformation
  which reflects a vector x through the origin
  of a hyperplane represented by its normal vector
  v of unit length.

  The reflection can be viewed as a linear transformation
  represented by a matrix P, i.e. `x = Px`.

  Args:
    x: A numpy array of shape (N, 1). The column
      vector to reflect.
    u: A numpy array of shape (N, 1). The unnormalized
      vector normal to the hyperplane.
    apply: bool indicating whether to apply the reflection
      directly and obtain x' or return the matrix P if set
      to `False`.

  Returns:
    refl: a numpy array of shape (N, 1) if apply is `True`.
    P: a numpy array of shape (N, N) if apply is `False`.
  """
  # grab dimension of column vector
  N = x.shape[0]

  # normalize u
  v = u / l2_norm(u)

  if apply:
    # compute projection of x onto v
    proj = np.dot(np.dot(v, v.T), x)

    # and finally reflect
    refl = x - 2*proj

    return refl
  else:
    P = np.eye(N) - 2*np.dot(v, v.T)

    return P


def normalize(x, inplace=False):
  """Normalize an input vector x.
  """
  norm_sq = x @ x
  if not np.isclose(norm_sq, 1.):
    norm = np.sqrt(norm_sq)
    if np.isclose(norm, 0):
      raise ZeroDivisionError("[!] Norm is very close to 0.")
    if inplace:
      x /= norm
    return x / norm


def l2_norm(x):
  """L2 or "euclidean" norm.
  """
  return np.sqrt(np.dot(x.T, x))


def inf_norm(x):
  """Infinity norm.
  """
  return np.max(np.abs(x))


def norm(x, p):
  """Returns the p norm of a vector.
  """
  v = np.array(x).flatten()

  error_msg = "x must be 1D"
  assert v.ndim == 1, error_msg
  error_msg = "p must be >= 1"
  assert p >= 1, error_msg

  N = v.shape[0]

  summer = KahanSum()
  for i in range(N):
    summer.add(np.power(np.abs(v[i]), p))

  return np.power(summer.cur_sum(), 1./p)


def sign(x):
  """Returns the sign of the variable x.
  """
  if x + 0 < 0:
    return -1
  return 1


def herm(A):
  """Returns the conjugate transpose of A.

  Equivalent to the H operator `A.H`.
  """
  return A.T.conj()


def upper_diag(A, diag=False):
  """Grabs the super-diagonal elements of a square matrix A.
  """
  m = len(A)
  U = np.zeros_like(A)

  for i in range(m):
    l_b = i + 1
    if diag:
      l_b = i
    for j in range(l_b, m):
      U[i, j] = A[i, j]

  return U


def lower_diag(A, diag=False):
  """Grabs the sub-diagonal elements of a square matrix A.
  """
  m = len(A)
  L = np.zeros_like(A)

  for i in range(m):
    u_b = i
    if diag:
      u_b = i + 1
    for j in range(0, u_b):
      L[i, j] = A[i, j]

  return L


def diag(A):
  """Grabs the diagonal elements of a square matrix A.
  """
  N = len(A)
  D = np.zeros([N, 1])

  for i in range(N):
    D[i] = A[i, i]

  return D


def create_diag(x):
  """Create a square matrix whose diagonal elements are the elements of x.
  """
  N = x.shape[0]
  D = np.zeros([N, N])

  for i in range(N):
    D[i, i] = x[i]

  return D


def unit_diag(A):
  """Fills the diagonal elements of a square matrix A with 1's.
  """
  m = len(A)

  for i in range(m):
    A[i, i] = 1

  return A


def multi_dot(l):
  """Performs consecutive dot products of the
  arrays in the list l from left to right.

  For example, given l = [A, B, C], returns
  `np.dot(np.dot(A, B), C)`.
  """
  return reduce(np.dot, l)


def basis_vec(k, n, flat=False):
  """Creates the k'th standard basis vector in R^n.
  """
  error_msg = "[!] k cannot exceed {}.".format(n)
  assert (k < n), error_msg

  b = np.zeros([n, 1])
  b[k] = 1
  if flat:
    return b.flatten()
  return b


def basis_arr(ks, n):
  """Creates an array of k'th standard basis vectors in R^n
  according to each k in ks.
  """
  error_msg = "[!] ks cannot exceed {}.".format(n)
  assert (np.max(ks) < n), error_msg

  b = np.zeros([n, n])
  for i, k in enumerate(ks):
    b[i, k] = 1
  return b


def random_spd(n):
  """Creates a random, symmetric, positive-definite matrix.

  Reference: https://math.stackexchange.com/a/358092/235367
  """
  A = np.random.rand(n, n)
  A = 0.5 * (A + A.T)
  A = A + n*np.eye(n)
  return A


def random_symmetric(n):
  """Creates a random, symmetric matrix.
  """
  A = np.random.randn(n, n)
  # alternatively, we could have done
  # A = A.T @ A but addition is cheaper
  # than matrix multiplication
  A = 0.5 * (A + A.T)
  return A


def is_symmetric(A):
  """Returns True if A is symmetric.
  """
  return np.allclose(A, A.T)
is_symm = is_symmetric


def is_square(A):
  """Returns True if A is square.
  """
  return A.shape[0] == A.shape[1]