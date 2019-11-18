import numpy as np
import os
#from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataLocation = input("Data Location: (default: ./output)")
if (dataLocation == ""):
    dataLocation = "./output"

printLocation = input("Plot Location: (default: ./plot)")
if (printLocation == ""):
    printLocation = "./plot"

def readLine(line):
    values = []
    for str in line.split("\t"):
        if (str.count(".") > 0):
            values.append(float(str))
        elif (str.count("nan") > 0):
            values.append(0)
        elif (str.count("inf") > 0):
            values.append(10)
        else:
            values.append(int(str))
    return values


def plotAndPrintData(fileName):
    f = open(dataLocation+"/"+fileName, "r")
    lines = f.readlines()

    """
    Expected format for the file is
    shape (ex: 2 1024)
    0	0	val
    ..
    i	j	val
    ..
    """

    shape = tuple(readLine(lines.pop(0)))
    Z = np.zeros(shape)
    for line in lines:
        values = readLine(line)
        z = values.pop()
        Z[tuple(values)] = z

    if (len(shape) == 2):
        X = np.linspace(1, shape[1], shape[1])
        plt.plot(X, Z[0, :], label='Prey')
        plt.plot(X, Z[1, :], label='Predator')
    elif (len(shape) == 3):
        X = np.outer(np.linspace(0, shape[1]-1, shape[1]), np.ones(shape[2]))
        Y = np.outer(np.ones(shape[1]), np.linspace(0, shape[2]-1, shape[2]))
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.plot_surface(X,Y,Z)

    else:
        return

    plt.savefig(printLocation+"/"+fileName.replace(".dat", ".png"))
    plt.close()


for file in os.listdir(printLocation):
    if ".png" in file:
        os.remove(printLocation+"/"+file)

for file in os.listdir(dataLocation):
    plotAndPrintData(file)

# ax = fig.add_subplot(111, projection='3d')
# x, y = np.array(2), np.array(128)
# X, Y = np.meshgrid(x, y)

# ax.plot_surface(X, Y, Z)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()
