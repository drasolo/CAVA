import cv2
import numpy as np


def _order(p):
    p = np.asarray(p, np.float32).reshape(4, 2)
    c = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    return np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)


def find_board(bgr):
    # Bright AND unsaturated: a pure brightness threshold also grabs light
    # coloured clutter touching the board and merges it into the contour.
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    th = cv2.inRange(hsv, (0, 0, 150), (179, 60, 255))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            return _order(approx)
    return _order(cv2.boxPoints(cv2.minAreaRect(c)))


def warp_board(bgr, corners, side):
    dst = np.float32([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]])
    M = cv2.getPerspectiveTransform(np.float32(corners), dst)
    return cv2.warpPerspective(bgr, M, (side, side))
