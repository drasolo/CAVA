"""Colour histograms as features

Course: Lab 11 (pixel-level features), Lab 2-3 (mosaic: matching by mean
colour), exam Ex1 (characters differ a lot in colour: hair, skin, clothes).
Often stronger than HOG on cartoons, and the two concatenated beat either.

TASK
  hsv_histogram(bgr, bins=(8, 8, 8)) -> 1D float64, length prod(bins), sums to 1.
      3D joint histogram over H (range 0..180), S (0..256), V (0..256), in
      (h, s, v) C-order when flattened. Bin of a value x in range [lo, hi) with
      b bins: floor((x - lo) * b / (hi - lo)).
  chi2_distance(a, b) -> 0.5 * sum((a - b)^2 / (a + b + 1e-10))

RULES  no cv2.calcHist, np.histogramdd.
HINT   Compute each channel's bin index, then one combined index
       (ih * bs + is) * bv + iv and np.bincount(..., minlength=...).
"""


def hsv_histogram(bgr, bins=(8, 8, 8)):
    raise NotImplementedError


def chi2_distance(a, b):
    raise NotImplementedError
