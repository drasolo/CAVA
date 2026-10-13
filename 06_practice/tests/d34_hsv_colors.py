from collections import Counter

import cv2
import numpy as np
from cava_practice.testing import check

HUES = {"red": [(-4, 4)], "orange": [(10, 16)], "yellow": [(24, 32)], "green": [(50, 75)], "blue": [(100, 118)]}


def patch(name, seed):
    r = np.random.default_rng(seed)
    if name == "white":
        hsv = np.array([r.uniform(0, 179), r.uniform(0, 35), r.uniform(170, 250)])
    else:
        lo, hi = HUES[name][0]
        hsv = np.array([r.uniform(lo, hi) % 180, r.uniform(140, 255), r.uniform(110, 245)])
    img = np.tile(hsv, (16, 16, 1))
    img[..., 0] = (img[..., 0] + r.normal(0, 1.2, (16, 16))) % 180
    img[..., 1:] += r.normal(0, 8, (16, 16, 2))
    img = np.clip(img, 0, [179, 255, 255]).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def test(m):
    hsv = np.zeros((1, 4, 3), np.uint8)
    hsv[0, :, 0] = [2, 177, 60, 110]
    hsv[0, :, 1:] = 220
    mask = np.asarray(m.red_mask(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)))
    check(mask.shape == (1, 4), f"red_mask must return an HxW mask, got shape {mask.shape}")
    check(mask[0, 0] > 0 and mask[0, 1] > 0, "red_mask must catch red at BOTH hue 2 and hue 177 (the wrap-around)")
    check(mask[0, 2] == 0 and mask[0, 3] == 0, "red_mask must not fire on green (60) or blue (110)")
    errors = Counter()
    names = ["red", "orange", "yellow", "green", "blue", "white"]
    for name in names:
        for seed in range(30):
            got = m.color_name(patch(name, seed + 31 * names.index(name)))
            if got != name:
                errors[(name, got)] += 1
    acc = 1 - sum(errors.values()) / 180
    check(acc >= 0.97, f"color_name accuracy {acc:.1%} < 97%. Confusions: {dict(errors)}")
