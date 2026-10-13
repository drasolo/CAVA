import cv2
import numpy as np
from cava_practice import synth
from cava_practice.testing import check, close, forbid


def test(m):
    forbid(m, ["cv2.add", "cv2.subtract", "cv2.absdiff"])
    img = np.array([[0, 5, 128, 250, 255]], np.uint8)
    up = m.add_brightness(img, 10)
    check(up.dtype == np.uint8, f"add_brightness must return uint8, got {up.dtype}")
    check(up.tolist() == [[10, 15, 138, 255, 255]], f"+10 gave {up.tolist()}: 250+10 must saturate at 255")
    check(m.add_brightness(img, -10).tolist() == [[0, 0, 118, 240, 245]], "-10 must saturate at 0")
    a = np.array([[10, 200]], np.uint8)
    b = np.array([[200, 10]], np.uint8)
    check(m.abs_diff(a, b).tolist() == [[190, 190]], f"abs_diff gave {m.abs_diff(a, b).tolist()}, expected [[190, 190]]")
    x, y = synth.random_image(30, 40, seed=1), synth.random_image(30, 40, seed=2)
    close(m.abs_diff(x, y), cv2.absdiff(x, y), 0, "abs_diff vs cv2.absdiff")
    close(m.mean_abs_diff(x, y), cv2.absdiff(x, y).mean(), 1e-9, "mean_abs_diff")
