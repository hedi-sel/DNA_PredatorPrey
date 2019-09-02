import numpy as np
import os
#from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataLocation="output/"
printLocation="ploter/"

def readLine(line):
    values = []
    for str in line.split("\t"):
        if (str.count(".") > 0):
            values.append(float(str))
        else:
            values.append(int(str))
    return values

def plotAndPrintData(fileName):
    f = open(dataLocation+fileName, "r")
    lines = f.readlines()

    """
    Expected format for the file is
    im  jm
    0	0	val
    ..
    i	j	val
    ..
    """

    shape = tuple(readLine(lines.pop(0)))
    Z = np.zeros(shape)
    for line in lines:
        values = readLine(line)
        Z[values[0]][values[1]] = values[2]

    X = np.linspace(0, 127, 128)

    #for i in range(0,shape[0]):
    plt.plot(X, Z[0, :],label = 'Prey')
    plt.plot(X, Z[1, :],label = 'Predator')
    plt.savefig(printLocation+fileName.replace(".dat",".png"))
    plt.close()

for file in os.listdir("./ploter"):
    if file != os.path.basename(__file__):
        os.remove("./ploter/"+file)

for file in os.listdir("./output"):
    if file != os.path.basename('empty'):
        plotAndPrintData(file)

# ax = fig.add_subplot(111, projection='3d')
# x, y = np.array(2), np.array(128)
# X, Y = np.meshgrid(x, y)

# ax.plot_surface(X, Y, Z)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()
