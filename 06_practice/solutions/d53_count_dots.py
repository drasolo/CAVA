import cv2
import numpy as np


def count_pips(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    m = int(0.06 * min(h, w))
    roi = cv2.GaussianBlur(gray[m:h - m, m:w - m], (5, 5), 0)
    _, ink = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    area_min = 0.004 * h * w
    return int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= area_min))
