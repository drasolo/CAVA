import cv2
import numpy as np
from cava_practice.testing import check, close, forbid


def test(m):
    forbid(m, ["getGaussianKernel", "GaussianBlur"])
    for k, s in [(3, 0.8), (5, 1.0), (9, 2.0), (15, 3.5)]:
        g1 = np.asarray(m.gaussian_1d(k, s)).ravel()
        close(g1, cv2.getGaussianKernel(k, s).ravel(), 1e-9, f"gaussian_1d({k}, {s})")
        g2 = np.asarray(m.gaussian_2d(k, s))
        ref = cv2.getGaussianKernel(k, s) @ cv2.getGaussianKernel(k, s).T
        close(g2, ref, 1e-9, f"gaussian_2d({k}, {s})")
        check(abs(g2.sum() - 1) < 1e-9, f"gaussian_2d({k}, {s}) must sum to 1, sums to {g2.sum()}")
    b = np.asarray(m.box_kernel(5))
    check(b.shape == (5, 5) and abs(b.sum() - 1) < 1e-9 and np.ptp(b) == 0, "box_kernel(5) must be 5x5, constant, sum 1")
