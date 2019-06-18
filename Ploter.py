import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


f = open("lat_0100.dat","r")
lines = f.readlines()
lines = np.zeroes(127,127)

def function(x, t):
    return x**2 + y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = t = np.linspace(-3.0, 3.0, 7)
X, T = np.meshgrid(x, y)
zs = np.array(function(np.ravel(X), np.ravel(T)))
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()