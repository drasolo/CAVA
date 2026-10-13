import cv2
import numpy as np


def resize_nearest(img, dsize):
    w, h = dsize
    H, W = img.shape[:2]
    ys = np.minimum((np.arange(h) * H) // h, H - 1)
    xs = np.minimum((np.arange(w) * W) // w, W - 1)
    return img[ys][:, xs]


def fit_square(img, side):
    H, W = img.shape[:2]
    s = side / max(H, W)
    nw, nh = max(1, round(W * s)), max(1, round(H * s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((side, side) + img.shape[2:], img.dtype)
    y0, x0 = (side - nh) // 2, (side - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out
