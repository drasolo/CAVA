import cv2
import numpy as np


def classify_cell(cell):
    h, w = cell.shape[:2]
    my, mx = int(h * 0.12), int(w * 0.12)
    roi = cv2.GaussianBlur(cell[my:h - my, mx:w - mx], (5, 5), 0)
    t, ink = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark, light = roi[roi <= t], roi[roi > t]
    if dark.size == 0 or light.size == 0 or light.mean() - dark.mean() < 60 or (ink > 0).mean() < 0.02:
        return "empty"
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, hier = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    min_hole = 0.02 * ink.size
    for i, c in enumerate(contours):
        if hier[0][i][3] != -1 and cv2.contourArea(c) > min_hole:
            return "0"
    return "X"
