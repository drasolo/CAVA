import numpy as np


def correlate2d(img, kernel):
    img = img.astype(np.float64)
    kh, kw = kernel.shape
    py, px = kh // 2, kw // 2
    pad = np.pad(img, ((py, py), (px, px)), mode="edge")
    H, W = img.shape
    out = np.zeros((H, W))
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * pad[i:i + H, j:j + W]
    return out


def convolve2d(img, kernel):
    return correlate2d(img, kernel[::-1, ::-1])
