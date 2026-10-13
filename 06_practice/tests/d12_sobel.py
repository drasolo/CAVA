import cv2
import numpy as np
from cava_practice import synth
from cava_practice.testing import check, close, forbid


def test(m):
    forbid(m, ["Sobel", "Scharr", "spatialGradient", "cartToPolar", "phase"])
    img = synth.smooth_image(50, 70, seed=5)
    gx, gy = m.sobel(img)
    close(gx, cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3), 1e-6, "gx")
    close(gy, cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3), 1e-6, "gy")
    ramp = np.tile(np.arange(20, dtype=np.uint8) * 5, (20, 1))
    gx, gy = m.sobel(ramp)
    check(gx[10, 10] > 0 and abs(gy[10, 10]) < 1e-9, "a left-to-right brightening ramp must give gx > 0, gy = 0")
    gx, gy = m.sobel(ramp.T.copy())
    check(gy[10, 10] > 0, "a top-to-bottom brightening ramp must give gy > 0 (y grows downward)")
    gx = np.array([[1.0, 0, -1, 0, 3]])
    gy = np.array([[0.0, 1, 0, -1, 4]])
    mag, ang = m.magnitude_orientation(gx, gy)
    close(mag, [[1, 1, 1, 1, 5]], 1e-9, "magnitude")
    close(ang, [[0, 90, 180, 270, np.degrees(np.arctan2(4, 3))]], 1e-6, "orientation (degrees, [0, 360))")
