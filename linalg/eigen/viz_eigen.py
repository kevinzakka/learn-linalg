"""Visualize eigen iteration algorithms.
"""

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

import numpy as np
import numpy.linalg as LA

from linalg import utils


def power_iteration(A, max_iter=1000):
  v = np.random.randn(A.shape[0])
  vs = [v.copy()]
  for i in range(max_iter):
    v = A @ v
    v /= utils.l2_norm(v)
    vs.append(v.copy())
  return vs


if __name__ == "__main__":
  M = np.random.randn(3, 3)
  M = M.T @ M
  
  num_iters = 10
  eigvecs = power_iteration(M, num_iters)

  us = [v[0] for v in eigvecs]
  vs = [v[1] for v in eigvecs]
  ws = [v[2] for v in eigvecs]

  uf = us[-1]
  vf = vs[-1]
  wf = ws[-1]

  def gen_vec():
    for i in range(num_iters-1):
      yield (i, us[i], vs[i], ws[i])

  fig = plt.figure()
  ax = p3.Axes3D(fig)
  iter_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

  def update_vec(iuvw):
    i, u, v, w = iuvw
    origin = [0, 0, 0]
    x, y, z = zip(origin, origin, origin)
    ax.quiver(x, y, z, u, v, w, length=0.05, normalize=True)
    iter_text.set_text('iter={}'.format(i))


  origin = [0, 0, 0]
  x, y, z = zip(origin, origin, origin)
  ax.quiver(x, y, z, uf, vf, wf, length=0.05, normalize=True, color='C1')
  vec_ani = animation.FuncAnimation(fig, update_vec, gen_vec, interval=1000, blit=False)
  vec_ani.save('./power_iteration.gif', writer='imagemagick', fps=2)
