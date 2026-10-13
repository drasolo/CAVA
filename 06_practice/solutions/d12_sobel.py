import cv2
import numpy as np

KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)


def sobel(gray):
    g = gray.astype(np.float64)
    return cv2.filter2D(g, cv2.CV_64F, KX), cv2.filter2D(g, cv2.CV_64F, KX.T)


def magnitude_orientation(gx, gy):
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 360.0
    return mag, ang
