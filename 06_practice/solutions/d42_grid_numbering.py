import numpy as np


def cell_boxes(side, n, margin=0.0):
    edges = [round(k * side / n) for k in range(n + 1)]
    out = []
    for r in range(n):
        for c in range(n):
            x0, x1, y0, y1 = edges[c], edges[c + 1], edges[r], edges[r + 1]
            mx, my = round((x1 - x0) * margin), round((y1 - y0) * margin)
            out.append((x0 + mx, y0 + my, x1 - mx, y1 - my))
    return out


def number_map(n, scheme):
    if scheme == "row_major":
        return np.arange(1, n * n + 1).reshape(n, n)
    if scheme == "ring_cw":
        m = np.zeros((n, n), int)
        path = [(0, c) for c in range(n)]
        path += [(r, n - 1) for r in range(1, n)]
        path += [(n - 1, c) for c in range(n - 2, -1, -1)]
        path += [(r, 0) for r in range(n - 2, 0, -1)]
        for k, (r, c) in enumerate(path, start=1):
            m[r, c] = k
        return m
    raise ValueError(scheme)


def qwirkle_pos(row, col):
    return f"{row + 1}{chr(ord('A') + col)}"


def parse_qwirkle_pos(s):
    s = s.strip()
    return int(s[:-1]) - 1, ord(s[-1].upper()) - ord("A")
