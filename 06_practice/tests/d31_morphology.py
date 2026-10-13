import cv2
import numpy as np
from cava_practice.testing import check, forbid


def test(m):
    forbid(m, ["cv2.erode", "cv2.dilate", "morphologyEx", "scipy", "skimage"])
    r = np.random.default_rng(5)
    b = np.where(r.random((40, 50)) > 0.6, 255, 0).astype(np.uint8)
    b[10:25, 10:30] = 255
    for k in (3, 5):
        se = np.ones((k, k), np.uint8)
        pairs = [
            ("erode", m.erode(b, k), cv2.erode(b, se)),
            ("dilate", m.dilate(b, k), cv2.dilate(b, se)),
            ("opening", m.opening(b, k), cv2.morphologyEx(b, cv2.MORPH_OPEN, se)),
            ("closing", m.closing(b, k), cv2.morphologyEx(b, cv2.MORPH_CLOSE, se)),
        ]
        for name, got, ref in pairs:
            got = np.asarray(got)
            check(got.shape == ref.shape, f"{name}(k={k}) shape {got.shape}")
            bad = int((got != ref).sum())
            check(bad == 0, f"{name}(k={k}) differs from OpenCV on {bad} pixels (border handling?)")
    check(m.ANSWERS == {"remove_white_specks": "opening", "fill_black_holes": "closing"},
          "ANSWERS: erosion shrinks white, dilation grows white. Which order kills specks, which fills holes?")
