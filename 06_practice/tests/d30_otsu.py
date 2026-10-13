import cv2
import numpy as np
from cava_practice import synth
from cava_practice.testing import check, forbid


def test(m):
    forbid(m, ["cv2.threshold", "skimage"])
    imgs = [synth.symbol_cell(k, 120, seed=s) for k, s in [("X", 1), ("0", 2), ("X", 3)]]
    imgs.append(synth.smooth_image(80, 80, seed=11))
    r = np.random.default_rng(3)
    bimodal = np.clip(np.r_[r.normal(60, 12, 3000), r.normal(170, 20, 2000)], 0, 255).astype(np.uint8).reshape(50, 100)
    imgs.append(bimodal)
    for i, img in enumerate(imgs):
        ref, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = m.otsu_threshold(img)
        check(abs(int(t) - int(ref)) <= 1, f"image {i}: Otsu threshold {t}, OpenCV says {int(ref)}")
    img = imgs[0]
    b = m.binarize(img, 120)
    check(b.dtype == np.uint8 and set(np.unique(b)) <= {0, 255}, "binarize must return uint8 with only 0 and 255")
    check(np.array_equal(b > 0, img > 120), "binarize: 255 where gray > t")
    check(np.array_equal(m.binarize(img, 120, invert=True) > 0, img <= 120), "binarize(invert=True): 255 where gray <= t")
