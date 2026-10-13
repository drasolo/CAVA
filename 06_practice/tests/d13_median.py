import cv2
import numpy as np
from cava_practice import synth
from cava_practice.testing import check, forbid


def test(m):
    forbid(m, ["medianBlur", "scipy"])
    img = synth.smooth_image(60, 80, seed=6)
    noisy = img.copy()
    r = np.random.default_rng(1)
    idx = r.random(img.shape)
    noisy[idx < 0.05] = 0
    noisy[idx > 0.95] = 255
    for k in (3, 5):
        got = m.median_filter(noisy, k)
        check(got.shape == img.shape and got.dtype == np.uint8, f"median_filter must return uint8 {img.shape}")
        ref = cv2.medianBlur(noisy, k)
        p = k // 2
        diff = np.abs(got[p:-p, p:-p].astype(int) - ref[p:-p, p:-p].astype(int))
        check(diff.max() <= 1, f"k={k}: interior differs from cv2.medianBlur by up to {diff.max()}")
    check(m.CHOICE == {"salt_pepper": "median", "gaussian_noise": "gaussian"},
          "CHOICE: think about which filter ignores outliers and which averages small noise away")
