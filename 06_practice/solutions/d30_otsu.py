import numpy as np


def otsu_threshold(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    w0 = np.cumsum(p)
    m = np.cumsum(p * np.arange(256))
    mt = m[-1]
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 - m) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = 0
    return int(np.argmax(between))


def binarize(gray, t, invert=False):
    fg = gray <= t if invert else gray > t
    return np.where(fg, 255, 0).astype(np.uint8)
