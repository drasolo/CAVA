import numpy as np


def to_gray(img_bgr):
    b, g, r = [img_bgr[..., i].astype(np.float64) for i in range(3)]
    return np.round(0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def bgr_to_rgb(img):
    return img[..., ::-1].copy()
