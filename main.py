import matplotlib.pyplot as plt
import skimage.data
import skimage.color
import numpy as np
import skimage.filters
import skimage.segmentation
from skimage import io
import scipy.ndimage
import helper

# TODO

# 1. EXTRACT control points from the B-Spline curve
# 2. NORMALIZE (BETWEEN 0 AND 1)

# FOLDERS
    # Parkinsons
    # Normal
    # SWEDD

def segment():
    image = io.imread("brain.jpeg")
    plt.imshow(image,cmap="gray")

    image2 = skimage.filters.gaussian(image, 6.0)

    # Ask the user to select points on the image
    from scipy.interpolate import splprep, splev
    points = plt.ginput(n=0, timeout=0)
    points.append(points[0])

    # Convert the points to x and y arrays
    x, y = zip(*points)
    # Create the initial contour
    init1 = np.array([x, y]).T

    tck, u = splprep([x, y], s=0)

    # Generate a smooth curve from the spline
    s = np.linspace(0, 1, 400)
    curved_init = splev(s, tck)

    # Transpose the curved_init to match the format of the init array
    init1 = np.array(curved_init).T

    snakeContour1 = helper.kassSnake(image2, init1, wLine=0, wEdge=1.0, alpha=0.1, beta=0.1, gamma=0.001,
                                    maxIterations=5, maxPixelMove=None, convergence=0.1)

    plt.imshow(image, cmap="gray")
    image2 = skimage.filters.gaussian(image, 6.0)
    points = plt.ginput(n=0, timeout=0)
    points.append(points[0])
    x, y = zip(*points)
    init2 = np.array([x, y]).T
    tck, u = splprep([x, y], s=0)
    s = np.linspace(0, 1, 400)
    curved_init = splev(s, tck)
    init2 = np.array(curved_init).T

    snakeContour2 = helper.kassSnake(image2, init2, wLine=0, wEdge=1.0, alpha=0.1, beta=0.1, gamma=0.001,
                                    maxIterations=5, maxPixelMove=None, convergence=0.1)


    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.plot(init1[:, 0], init1[:, 1], '--r', lw=2)
    plt.plot(snakeContour1[:, 0], snakeContour1[:, 1], '-b', lw=2)
    plt.plot(init2[:, 0], init2[:, 1], '--r', lw=2)
    plt.plot(snakeContour2[:, 0], snakeContour2[:, 1], '-b', lw=2)
    plt.show()


if __name__ == '__main__':
    segment()