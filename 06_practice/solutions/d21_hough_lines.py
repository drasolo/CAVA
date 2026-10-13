import numpy as np


def hough_accumulator(edges, theta_step=1.0, rho_step=1.0):
    H, W = edges.shape
    D = int(np.ceil(np.hypot(H, W)))
    thetas = np.arange(0, 180, theta_step)
    rhos = np.arange(-D, D + rho_step, rho_step)
    ys, xs = np.nonzero(edges)
    t = np.deg2rad(thetas)
    rho = xs[:, None] * np.cos(t) + ys[:, None] * np.sin(t)
    ri = np.round((rho + D) / rho_step).astype(int)
    ti = np.broadcast_to(np.arange(len(thetas)), ri.shape)
    acc = np.zeros((len(rhos), len(thetas)), np.int64)
    np.add.at(acc, (ri.ravel(), ti.ravel()), 1)
    return acc, rhos, thetas


def top_lines(acc, rhos, thetas, k, nms=5):
    a = acc.astype(np.float64).copy()
    out = []
    for _ in range(k):
        i, j = np.unravel_index(np.argmax(a), a.shape)
        out.append((float(rhos[i]), float(thetas[j])))
        a[max(0, i - nms):i + nms + 1, max(0, j - nms):j + nms + 1] = 0
    return out
