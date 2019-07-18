import numpy as np
#from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataLocation="output/"
printLocation="ploter/"
dataExtension=".dat"

def readLine(line):
    values = []
    for str in line.split("\t"):
        if (str.count(".") > 0):
            values.append(float(str))
        else:
            values.append(int(str))
    return values

def plotAndPrintData(fileName):
    f = open(dataLocation+fileName+dataExtension, "r")
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

    def function(x, y):
        return Z[x, y]

    X = np.linspace(0, 127, 128)

    #for i in range(0,shape[0]):
    plt.plot(X, Z[0, :],label = 'Prey')
    plt.plot(X, Z[1, :],label = 'Predator')
    plt.savefig(printLocation+fileName+".jpg")
    plt.close()

plotAndPrintData("lat_0")
plotAndPrintData("lat_20")
plotAndPrintData("lat_40")
plotAndPrintData("lat_60")
# ax = fig.add_subplot(111, projection='3d')
# x, y = np.array(2), np.array(128)
# X, Y = np.meshgrid(x, y)

# ax.plot_surface(X, Y, Z)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()
