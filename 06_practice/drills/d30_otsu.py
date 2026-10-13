"""Otsu's threshold from scratch

Course: Curs 6 (binary images). Otsu picks the threshold that best separates
the histogram into two classes (max between-class variance). You'll use it to
binarize cells, tiles, and dots under changing light.

TASK
  otsu_threshold(gray) -> int t in 0..255
      Pixels <= t are class 0, pixels > t are class 1 (OpenCV's convention).
      Choose t maximising w0 * w1 * (mu0 - mu1)^2.
  binarize(gray, t, invert=False) -> uint8 0/255
      255 where gray > t (or where gray <= t when invert=True).

RULES  no cv2.threshold, skimage.filters.
HINT   hist = np.bincount(gray.ravel(), minlength=256); use cumulative sums so
       every candidate t costs O(1).
"""


def otsu_threshold(gray):
    raise NotImplementedError


def binarize(gray, t, invert=False):
    raise NotImplementedError
