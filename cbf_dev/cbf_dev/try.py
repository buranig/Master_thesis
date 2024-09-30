#!/usr/bin/python
from scipy.spatial import HalfspaceIntersection

import numpy as np

halfspaces = np.array([[0.2, -0.59146353, 0.16887748],
                       [0., 1., -0.10081619],
                       [0., -1., -0.10081619],
                       [ 1., 0., -0.45],
                       [-1., 0., -0.45]])

halfspaces = np.array([[9.29002170e-01, 4.68910387e-08, -1.28444573e-07],
                       [-2.00000000e-01, 4.48484313e-04, -2.19472872e+01],
                       [0., 1., -0.10081619],
                       [0., -1., -0.10081619],
                       [ 1., 0., -0.45],
                       [-1., 0., -0.45]])

feasible_point = np.array([0.5, 0.5])

# hs = HalfspaceIntersection(halfspaces)

import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1, aspect='equal')

xlim, ylim = (-1, 3), (-1, 3)

ax.set_xlim(xlim)

ax.set_ylim(ylim)

x = np.linspace(-1, 3, 100)

symbols = ['-', '+', 'x', '*']

signs = [0, 0, -1, -1]

fmt = {"color": None, "edgecolor": "b", "alpha": 0.5}

for h, sym, sign in zip(halfspaces, symbols, signs):

    hlist = h.tolist()

    fmt["hatch"] = sym

    if h[1]== 0:

        ax.axvline(-h[2]/h[0], label='{}x+{}y+{}=0'.format(*hlist))

        xi = np.linspace(xlim[sign], -h[2]/h[0], 100)

        ax.fill_between(xi, ylim[0], ylim[1], **fmt)

    else:

        ax.plot(x, (-h[2]-h[0]*x)/h[1], label='{}x+{}y+{}=0'.format(*hlist))

        ax.fill_between(x, (-h[2]-h[0]*x)/h[1], ylim[sign], **fmt)

# x, y = zip(*hs.intersections)

# ax.plot(x, y, 'o', markersize=8)
# plt.xlim(-3,3)
# plt.ylim(-3,3)
plt.show()