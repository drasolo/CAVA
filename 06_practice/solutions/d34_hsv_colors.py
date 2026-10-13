import cv2
import numpy as np


def red_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = cv2.inRange(hsv, (0, 80, 60), (8, 255, 255))
    hi = cv2.inRange(hsv, (170, 80, 60), (179, 255, 255))
    return cv2.bitwise_or(lo, hi)


def color_name(bgr_patch):
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
    h, s, v = np.median(hsv, axis=0)
    if s < 60:
        return "white"
    # rotate hue so red is contiguous: 170..179 -> -10..-1
    hues = hsv[:, 0]
    hues = np.where(hues >= 165, hues - 180, hues)
    h = np.median(hues)
    if h < 7:
        return "red"
    if h < 20:
        return "orange"
    if h < 40:
        return "yellow"
    if h < 90:
        return "green"
    return "blue"
