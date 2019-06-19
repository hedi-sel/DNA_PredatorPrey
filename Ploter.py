import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


f = open("lat_1000.dat","r")
lines = f.readlines()
shape = (128,128)
Z1 = np.zeros(shape)
Z2 = np.zeros(shape)
for line in lines:
    values = line.split("\t")
    if len(values) == 4:
        Z1[int(values[0])][int(values[1])] = float(values[2])
        Z2[int(values[0])][int(values[1])] = float(values[3])

def function(x, t):
    return Y1[x,t]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = t = np.array(range(128))
X, T = np.meshgrid(x, t)

ax.plot_surface(X, T, Y1)

ax.set_xlabel('X Label')
ax.set_ylabel('T Label')
ax.set_zlabel('Z Label')

plt.show()