import cv2
import numpy as np


def hsv_histogram(bgr, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int64)
    bh, bs, bv = bins
    ih = hsv[:, 0] * bh // 180
    is_ = hsv[:, 1] * bs // 256
    iv = hsv[:, 2] * bv // 256
    idx = (ih * bs + is_) * bv + iv
    h = np.bincount(idx, minlength=bh * bs * bv).astype(np.float64)
    return h / h.sum()


def chi2_distance(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.5 * np.sum((a - b) ** 2 / (a + b + 1e-10))
