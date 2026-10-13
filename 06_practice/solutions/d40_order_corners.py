import numpy as np


def order_corners(pts):
    p = np.asarray(pts, np.float32).reshape(4, 2)
    c = p.mean(axis=0)
    # Sorting by angle around the centroid gives a consistent cycle; with y
    # pointing down, increasing angle is clockwise on screen: TL -> TR -> BR -> BL.
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    # The cycle can start anywhere, so rotate it to begin at top-left (min x+y).
    start = int(np.argmin(p.sum(axis=1)))
    return np.roll(p, -start, axis=0).astype(np.float32)
