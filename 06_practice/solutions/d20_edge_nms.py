import numpy as np


def nonmax_suppression(mag, ang):
    H, W = mag.shape
    a = ang % 180.0
    out = np.zeros_like(mag, dtype=np.float64)
    c = mag[1:-1, 1:-1]
    def sh(dy, dx):
        return mag[1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx]
    aa = a[1:-1, 1:-1]
    bins = [
        ((aa < 22.5) | (aa >= 157.5), (0, 1), (0, -1)),
        ((aa >= 22.5) & (aa < 67.5), (1, 1), (-1, -1)),
        ((aa >= 67.5) & (aa < 112.5), (1, 0), (-1, 0)),
        ((aa >= 112.5) & (aa < 157.5), (1, -1), (-1, 1)),
    ]
    keep = np.zeros_like(c, dtype=bool)
    for sel, n1, n2 in bins:
        keep |= sel & (c >= sh(*n1)) & (c >= sh(*n2))
    out[1:-1, 1:-1] = np.where(keep, c, 0)
    return out
