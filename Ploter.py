import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


f = open("lat_0000.dat", "r")
lines = f.readlines()

def readLine(line):
    values = []
    for str in line.split("\t"):
        if (str.count(".") > 0):
            values.append(float(str))
        else:
            values.append(int(str))
    return values

"""
Expected format for the file is
n	m
0	0	value
..
i	j	value
..
"""


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
