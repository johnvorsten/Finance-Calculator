""""""
# Python imports

# Third party imports
import matplotlib as mpl
import numpy as np
from scipy.misc import comb
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Local imports

# Declarations

#%%
def bernstein_poly(i, n, t):
    return comb(n, i) * (t**(n - i)) * (1 - t)**i


def generate_points_bezier_curve(points, nTimes=1000):
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])
    zPoints = np.array([p[2] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array(
        [bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    zvals = np.dot(zPoints, polynomial_array)

    return xvals, yvals, zvals

def plot_points_bezier_curve():
    nPoints = 4
    points = np.random.rand(nPoints, 3) * 200
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    zpoints = [p[2] for p in points]

    xvals, yvals, zvals = bezier_curve(points, nTimes=1000)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xvals, yvals, zvals, label='bezier')
    ax.plot(xpoints, ypoints, zpoints, "ro")
    for nr in range(len(points)):
        ax.text(points[nr][0], points[nr][1], points[nr][2], nr)

    plt.show()
    return None

if __name__ == "__main__":
    plot_points_bezier_curve()