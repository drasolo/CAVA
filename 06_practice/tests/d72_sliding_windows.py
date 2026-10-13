import math

import cv2
import numpy as np
from cava_practice import synth
from cava_practice.testing import check


def iou(a, b):
    xa, ya, xb, yb = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    area = lambda r: (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
    return inter / (area(a) + area(b) - inter)


def test(m):
    b = np.asarray(m.window_boxes(100, 120, 40, 20, [1.0]))
    check(b.shape == (4 * 5, 4), f"100x120 image, 40px window, step 20, scale 1: 4 rows x 5 cols = 20 windows, got {b.shape[0]}")
    check(tuple(b[0]) == (0, 0, 39, 39) and tuple(b[1]) == (20, 0, 59, 39), f"first windows wrong: {b[:2].tolist()} (rows outer, x inner)")
    b = np.asarray(m.window_boxes(100, 120, 40, 20, [0.5]))
    check(b.shape[0] == 1 * 2, f"at scale 0.5 the image is 50x60: 1 x 2 windows, got {b.shape[0]}")
    check(tuple(b[1]) == (40, 0, 119, 79), f"scale 0.5: window at x=20 maps back to (40, 0, 119, 79), got {tuple(b[1])}")
    b = np.asarray(m.window_boxes(90, 90, 64, 8, [1.0, 0.8, 1.25]))
    n = sum(((round(90 * s) - 64) // 8 + 1) ** 2 for s in (1.0, 0.8, 1.25) if round(90 * s) >= 64)
    check(b.shape[0] == n, f"multi-scale window count {b.shape[0]}, expected {n}")

    r = np.random.default_rng(4)
    tpl = synth.symbol_cell("0", 32, seed=7)
    for trial in range(4):
        img = synth.smooth_image(240, 320, seed=40 + trial) // 3 + 60
        size = int(r.integers(40, 110))
        x0, y0 = int(r.integers(0, 320 - size)), int(r.integers(0, 240 - size))
        img[y0:y0 + size, x0:x0 + size] = cv2.resize(tpl, (size, size))
        truth = (x0, y0, x0 + size - 1, y0 + size - 1)
        scales = [32 / s for s in range(36, 120, 6)]
        got = m.find_square(img, tpl, scales, 2)
        check(got is not None and len(got) == 4, f"find_square must return a box, got {got!r}")
        check(iou(got, truth) >= 0.6, f"trial {trial}: template pasted at {truth} (size {size}); you returned {tuple(got)}, IoU {iou(got, truth):.2f}")
