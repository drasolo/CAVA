"""Build a Gaussian kernel

Course: Curs 2 (smoothing). The Gaussian is the blur you'll reach for first in
every pipeline. sigma controls the strength; ksize must be big enough
(rule of thumb: ksize ~ 6*sigma, odd).

TASK
  gaussian_1d(ksize, sigma) -> 1D float64 array of length ksize, sums to 1.
      g[i] proportional to exp(-(i - c)^2 / (2 sigma^2)), c = (ksize-1)/2.
  gaussian_2d(ksize, sigma) -> ksize x ksize, sums to 1.
      Use separability: the 2D Gaussian is the outer product of two 1D ones.
  box_kernel(ksize)         -> ksize x ksize mean filter.

RULES  no cv2.getGaussianKernel.
"""


def gaussian_1d(ksize, sigma):
    raise NotImplementedError


def gaussian_2d(ksize, sigma):
    raise NotImplementedError


def box_kernel(ksize):
    raise NotImplementedError
