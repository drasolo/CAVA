import math

import cv2
import numpy as np


def window_boxes(img_h, img_w, win, step, scales):
    out = []
    for s in scales:
        H, W = round(img_h * s), round(img_w * s)
        for y in range(0, H - win + 1, step):
            for x in range(0, W - win + 1, step):
                out.append((math.floor(x / s), math.floor(y / s),
                            math.floor((x + win) / s) - 1, math.floor((y + win) / s) - 1))
    return np.array(out, int).reshape(-1, 4)


def find_square(img, template, scales, step):
    win = template.shape[0]
    best, best_box = -2.0, None
    for s in scales:
        H, W = round(img.shape[0] * s), round(img.shape[1] * s)
        if H < win or W < win:
            continue
        small = cv2.resize(img, (W, H))
        r = cv2.matchTemplate(small.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        sub = r[::step, ::step]
        iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
        if sub[iy, ix] > best:
            y, x = iy * step, ix * step
            best = sub[iy, ix]
            best_box = (math.floor(x / s), math.floor(y / s), math.floor((x + win) / s) - 1, math.floor((y + win) / s) - 1)
    return best_box
