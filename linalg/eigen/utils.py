from linalg.eigen.multi import eig


def is_spd(A):
  """Returns True if A is symmetric positive definite.
  """
  symm_cond = is_symmetric(A)
  eigvals, = eig(A, sort=False)
  poseig_cond = np.all(eigvals >= 0)
  return symm_cond and poseig_cond