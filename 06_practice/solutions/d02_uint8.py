import numpy as np


def add_brightness(img, delta):
    return np.clip(img.astype(np.int32) + delta, 0, 255).astype(np.uint8)


def abs_diff(a, b):
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)


def mean_abs_diff(a, b):
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))
