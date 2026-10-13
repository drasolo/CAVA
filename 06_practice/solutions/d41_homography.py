import numpy as np


def homography(src, dst):
    A, b = [], []
    for (x, y), (u, v) in zip(np.asarray(src, float), np.asarray(dst, float)):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b += [u, v]
    h = np.linalg.solve(np.array(A), np.array(b))
    return np.append(h, 1.0).reshape(3, 3)


def apply_h(H, pts):
    p = np.asarray(pts, float).reshape(-1, 2)
    hom = np.c_[p, np.ones(len(p))] @ H.T
    return hom[:, :2] / hom[:, 2:3]
