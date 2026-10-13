import cv2
import numpy as np
from cava_practice import synth
from cava_practice.testing import arr, check, close, forbid


def test(m):
    forbid(m, ["cvtColor"])
    red = np.zeros((4, 4, 3), np.uint8)
    red[..., 2] = 255
    g = arr(m.to_gray(red), "to_gray")
    check(g.dtype == np.uint8, f"to_gray must return uint8, got {g.dtype}")
    check(g.shape == (4, 4), f"to_gray shape {g.shape}, expected (4, 4)")
    check(int(g[0, 0]) == 76, f"pure red (BGR 0,0,255) should give 76, got {g[0, 0]}: channel order mixed up?")
    for seed in range(3):
        img = synth.random_image(37, 53, seed=seed)
        close(m.to_gray(img), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, "to_gray vs cv2")
    img = synth.random_image(5, 7, seed=9)
    rgb = arr(m.bgr_to_rgb(img), "bgr_to_rgb")
    check(np.array_equal(rgb, img[..., ::-1]), "bgr_to_rgb: channels not reversed")
