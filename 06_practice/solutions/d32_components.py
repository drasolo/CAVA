from collections import deque

import numpy as np

N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = N4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def label_components(binary, connectivity):
    fg = np.asarray(binary) != 0
    H, W = fg.shape
    labels = np.zeros((H, W), np.int32)
    nbrs = N4 if connectivity == 4 else N8
    n = 0
    for y0, x0 in zip(*np.nonzero(fg)):
        if labels[y0, x0]:
            continue
        n += 1
        labels[y0, x0] = n
        q = deque([(y0, x0)])
        while q:
            y, x = q.popleft()
            for dy, dx in nbrs:
                v, u = y + dy, x + dx
                if 0 <= v < H and 0 <= u < W and fg[v, u] and not labels[v, u]:
                    labels[v, u] = n
                    q.append((v, u))
    return n, labels


def component_areas(labels, n):
    return np.bincount(labels.ravel(), minlength=n + 1)[1:].tolist()
