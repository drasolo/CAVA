import numpy as np


def gaussian_1d(ksize, sigma):
    x = np.arange(ksize) - (ksize - 1) / 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def gaussian_2d(ksize, sigma):
    g = gaussian_1d(ksize, sigma)
    return np.outer(g, g)


def box_kernel(ksize):
    return np.full((ksize, ksize), 1.0 / ksize ** 2)
