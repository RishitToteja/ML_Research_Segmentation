import numpy as np
import scipy.linalg
import scipy.ndimage
import skimage
import skimage.filters
import scipy.interpolate


def kassSnake(image, initialContour, edgeImage=None, alpha=0.01, beta=0.1, wLine=0, wEdge=1, gamma=0.01,
              maxPixelMove=None, maxIterations=2500, convergence=0.1):
    maxIterations = int(maxIterations)
    if maxIterations <= 0:
        raise ValueError('maxIterations should be greater than 0.')

    convergenceOrder = 10

    image = skimage.img_as_float(image)
    isMultiChannel = image.ndim == 3

    if edgeImage is None and wEdge != 0:

        edgeImage = np.sqrt(scipy.ndimage.sobel(image, axis=0, mode='reflect') ** 2 +
                            scipy.ndimage.sobel(image, axis=1, mode='reflect') ** 2)

        edgeImage = (edgeImage - edgeImage.min()) / (edgeImage.max() - edgeImage.min())
    elif edgeImage is None:
        edgeImage = 0

    if isMultiChannel:
        externalEnergy = wLine * np.sum(image, axis=2) + wEdge * np.sum(edgeImage, axis=2)
    else:
        externalEnergy = wLine * image + wEdge * edgeImage

    externalEnergyInterpolation = scipy.interpolate.RectBivariateSpline(np.arange(externalEnergy.shape[1]),
                                                                        np.arange(externalEnergy.shape[0]),
                                                                        externalEnergy.T, kx=2, ky=2, s=0)

    x, y = initialContour[:, 0].astype(float), initialContour[:, 1].astype(float)

    previousX = np.empty((convergenceOrder, len(x)))
    previousY = np.empty((convergenceOrder, len(y)))

    n = len(x)
    r = 2 * alpha + 6 * beta
    q = -alpha - 4 * beta
    p = beta

    A = r * np.eye(n) + \
        q * (np.roll(np.eye(n), -1, axis=0) + np.roll(np.eye(n), -1, axis=1)) + \
        p * (np.roll(np.eye(n), -2, axis=0) + np.roll(np.eye(n), -2, axis=1))

    AInv = scipy.linalg.inv(A + gamma * np.eye(n))

    for i in range(maxIterations):

        fx = externalEnergyInterpolation(x, y, dx=1, grid=False)
        fy = externalEnergyInterpolation(x, y, dy=1, grid=False)

        xNew = np.dot(AInv, gamma * x + fx)

        yNew = np.dot(AInv, gamma * y + fy)
        if maxPixelMove:
            dx = maxPixelMove * np.tanh(xNew - x)
            dy = maxPixelMove * np.tanh(yNew - y)

            x += dx
            y += dy
        else:
            x = xNew
            y = yNew

        j = i % (convergenceOrder + 1)

        if j < convergenceOrder:
            previousX[j, :] = x
            previousY[j, :] = y
        else:
            distance = np.min(np.max(np.abs(previousX - x[None, :]) + np.abs(previousY - y[None, :]), axis=1))

            if distance < convergence:
                break

    print('Finished at', i)

    return np.array([x, y]).T