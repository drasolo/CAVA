import importlib.util
from pathlib import Path

import cv2
import numpy as np
from cava_practice.testing import check

KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)


def grad(img):
    gx = cv2.filter2D(img, cv2.CV_64F, KX)
    gy = cv2.filter2D(img, cv2.CV_64F, KX.T)
    return np.hypot(gx, gy), np.degrees(np.arctan2(gy, gx)) % 360


def soft_edge(coord):
    return 255.0 / (1 + np.exp(-coord / 2.0))


def strong_per_row(out):
    s = out > 0.5 * out.max()
    return [int(s[y].sum()) for y in range(5, out.shape[0] - 5)]


def test(m):
    H = W = 60
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    cases = {
        "vertical edge": soft_edge(x - 30.2),
        "horizontal edge (check rows by transposing)": soft_edge(y - 30.2).T.copy(),
        "diagonal edge": soft_edge((x + y - 60.3) / np.sqrt(2)),
        "anti-diagonal edge": soft_edge((x - y - 0.3) / np.sqrt(2)),
    }
    for name, img in cases.items():
        mag, ang = grad(img)
        before = strong_per_row(mag)
        out = np.asarray(m.nonmax_suppression(mag, ang), np.float64)
        check(out.shape == mag.shape, f"{name}: output shape {out.shape}, expected {mag.shape}")
        check(np.all(out <= mag + 1e-9), f"{name}: NMS may only keep or zero values, never increase them")
        check(not out[0].any() and not out[-1].any() and not out[:, 0].any() and not out[:, -1].any(), "set the border to 0")
        rows = strong_per_row(out)
        check(max(before) > 2, "test setup: edge should start thick")
        # On a pixel grid, diagonal neighbours skip every other pixel, so a 45 degree
        # edge keeps 2 pixels per row. Real Canny interpolates to avoid this.
        allowed = 2 if "diagonal" in name else 1
        check(all(1 <= r <= allowed for r in rows),
              f"{name}: each row should keep {allowed} strong pixel(s) after NMS, got counts {sorted(set(rows))} "
              "(check the neighbour directions: y grows DOWN)")
