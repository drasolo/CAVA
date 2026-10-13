import math

import numpy as np


def ransac_iterations(p, r, k):
    return math.ceil(math.log(1 - p) / math.log(1 - r ** k))


def fit_line_ransac(points, thresh, n_iter, seed=0):
    pts = np.asarray(points, np.float64)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_iter):
        i, j = rng.choice(len(pts), 2, replace=False)
        d = pts[j] - pts[i]
        norm = np.hypot(*d)
        if norm == 0:
            continue
        n = np.array([-d[1], d[0]]) / norm
        c = -n @ pts[i]
        mask = np.abs(pts @ n + c) < thresh
        if best is None or mask.sum() > best.sum():
            best = mask
    inl = pts[best]
    mu = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - mu)
    n = vt[-1]
    c = -n @ mu
    mask = np.abs(pts @ n + c) < thresh
    return (float(n[0]), float(n[1]), float(c)), mask
