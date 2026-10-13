import numpy as np

CHOICE = {"salt_pepper": "median", "gaussian_noise": "gaussian"}


def median_filter(gray, k):
    p = k // 2
    pad = np.pad(gray, p, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, (k, k))
    return np.median(win, axis=(-2, -1)).astype(np.uint8)
