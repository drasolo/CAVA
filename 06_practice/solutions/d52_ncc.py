import numpy as np


def match_ncc(img, tpl):
    I = np.asarray(img, np.float64)
    T = np.asarray(tpl, np.float64)
    h, w = T.shape
    T = T - T.mean()
    tn = np.sqrt((T ** 2).sum())
    win = np.lib.stride_tricks.sliding_window_view(I, (h, w))
    P = win - win.mean(axis=(-2, -1), keepdims=True)
    num = (P * T).sum(axis=(-2, -1))
    den = np.sqrt((P ** 2).sum(axis=(-2, -1))) * tn
    return np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1), 0.0)


def best_match(img, tpl):
    r = match_ncc(img, tpl)
    y, x = np.unravel_index(np.argmax(r), r.shape)
    return int(x), int(y)
