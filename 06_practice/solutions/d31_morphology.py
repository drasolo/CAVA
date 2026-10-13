import numpy as np

ANSWERS = {"remove_white_specks": "opening", "fill_black_holes": "closing"}


def _window(binary, k, fill, op):
    p = k // 2
    pad = np.pad(binary, p, mode="constant", constant_values=fill)
    H, W = binary.shape
    out = pad[0:H, 0:W].copy()
    for i in range(k):
        for j in range(k):
            out = op(out, pad[i:i + H, j:j + W])
    return out


def erode(binary, k):
    return _window(binary, k, 255, np.minimum)


def dilate(binary, k):
    return _window(binary, k, 0, np.maximum)


def opening(binary, k):
    return dilate(erode(binary, k), k)


def closing(binary, k):
    return erode(dilate(binary, k), k)
