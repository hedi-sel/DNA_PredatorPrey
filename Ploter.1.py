import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import open


f = open(u"lat_0000.dat", u"r")
lines = f.readlines()

def readLine(line):
    values = []
    for unicode in line.split(u"\t"):
        if (unicode.count(u".") > 0):
            values.append(float(unicode))
        else:
            values.append(int(unicode))
    return values

u"""
Expected format for the file is
im  jm
0	0	val
..
i	j	val
..
"""


from __future__ import absolute_import
shape = tuple(readLine(lines.pop(0)))
Z = np.zeros(shape)
for line in lines:
    values = readLine(line)
    Z[values[0]][values[1]] = values[2]

def function(x, y):
    return Z[x, y]

X = np.linspace(0, 127, 128)

plt.plot(X, Z[1,:])
plt.show()
# ax = fig.add_subplot(111, projection='3d')
# x, y = np.array(2), np.array(128)
# X, Y = np.meshgrid(x, y)

# ax.plot_surface(X, Y, Z)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()
