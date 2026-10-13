import cv2
import numpy as np
from cava_practice import synth
from cava_practice.testing import check, close, forbid


def test(m):
    forbid(m, ["calcHist", "histogramdd"])
    for seed, bins in [(0, (8, 8, 8)), (1, (18, 4, 4)), (2, (6, 3, 2))]:
        img = synth.random_image(40, 30, seed=seed)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ref = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256]).ravel()
        ref = ref / ref.sum()
        got = np.asarray(m.hsv_histogram(img, bins), np.float64)
        check(got.ndim == 1 and got.size == np.prod(bins), f"bins {bins}: expected a flat vector of {np.prod(bins)}, got shape {got.shape}")
        close(got, ref, 1e-6, f"hsv_histogram bins={bins}")
    a = np.array([0.5, 0.5, 0])
    check(abs(m.chi2_distance(a, a)) < 1e-9, "chi2_distance(a, a) must be 0")
    close(m.chi2_distance(a, np.array([0, 0.5, 0.5])), 0.5, 1e-6, "chi2_distance([.5,.5,0], [0,.5,.5])")
