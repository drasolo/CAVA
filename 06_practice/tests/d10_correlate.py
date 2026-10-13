import cv2
import numpy as np
from cava_practice import synth
from cava_practice.testing import check, close, forbid


def test(m):
    forbid(m, ["filter2D", "sepFilter2D", "scipy"])
    img = synth.smooth_image(40, 60, seed=4).astype(np.float64)
    r = np.random.default_rng(0)
    for shape in [(3, 3), (5, 5), (3, 5), (7, 1)]:
        k = r.normal(size=shape)
        ref = cv2.filter2D(img, cv2.CV_64F, k, borderType=cv2.BORDER_REPLICATE)
        got = m.correlate2d(img, k)
        check(got.shape == img.shape, f"output must have the image's shape {img.shape}, got {got.shape}")
        close(got, ref, 1e-6, f"correlate2d with a {shape} kernel")
        ref_conv = cv2.filter2D(img, cv2.CV_64F, k[::-1, ::-1], borderType=cv2.BORDER_REPLICATE)
        close(m.convolve2d(img, k), ref_conv, 1e-6, f"convolve2d with a {shape} kernel")
    delta = np.zeros((7, 7))
    delta[3, 3] = 1
    k = np.arange(9.0).reshape(3, 3)
    resp = m.convolve2d(delta, k)
    check(np.allclose(resp[2:5, 2:5], k), "convolving an impulse must reproduce the kernel itself (that's the definition)")
