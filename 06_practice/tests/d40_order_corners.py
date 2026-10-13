import numpy as np
from cava_practice.testing import check


def test(m):
    r = np.random.default_rng(0)
    base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
    for trial in range(200):
        ang = np.deg2rad(r.uniform(-30, 30))
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        sx, sy = r.uniform(80, 300, 2)
        pts = (base * [sx, sy]) @ R.T + r.uniform(200, 600, 2) + r.normal(0, 8, (4, 2))
        perm = r.permutation(4)
        shuffled = pts[perm]
        if trial % 2:
            shuffled = shuffled.reshape(4, 1, 2)
        got = np.asarray(m.order_corners(shuffled))
        check(got.shape == (4, 2), f"return a 4x2 array, got shape {got.shape}")
        check(got.dtype == np.float32, f"return float32 (getPerspectiveTransform wants it), got {got.dtype}")
        check(np.allclose(got, pts, atol=1e-3),
              f"rotation {np.degrees(ang):.1f} deg: got {got.round(1).tolist()}, expected {pts.round(1).tolist()} (TL, TR, BR, BL)")
