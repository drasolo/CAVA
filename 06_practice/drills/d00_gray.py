"""Grayscale by hand, and the BGR trap

Course: Lab 1. cv2.imread returns BGR, not RGB. If you mix them up, red and
blue silently swap, and every colour threshold you write later is wrong.

TASK
  to_gray(img_bgr)  -> uint8 array HxW
      Y = 0.299*R + 0.587*G + 0.114*B, rounded to the nearest integer.
  bgr_to_rgb(img)   -> HxWx3 array with the channel order reversed.

RULES  no cv2.cvtColor.
HINTS  In a BGR image, img[..., 2] is RED. Compute in float64, then
       np.round(...).astype(np.uint8).
"""
import numpy as np


def to_gray(img_bgr):
    raise NotImplementedError


def bgr_to_rgb(img):
    raise NotImplementedError
