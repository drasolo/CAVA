"""The uint8 overflow trap

Course: everywhere. Images are uint8 (0..255). In numpy, uint8 250 + 10 == 4
and 10 - 20 == 246. Frame differencing (Qwirkle, background subtraction)
breaks silently if you forget this.

TASK
  add_brightness(img, delta) -> uint8, same shape. Saturate at 0 and 255.
  abs_diff(a, b)            -> uint8 |a - b| per pixel, correct in both orders.
  mean_abs_diff(a, b)       -> float, mean of |a - b|.

RULES  no cv2.add, cv2.subtract, cv2.absdiff.
HINTS  hidden on purpose: python check.py d02 --hint (one step at a time)
"""


def add_brightness(img, delta):
    raise NotImplementedError


def abs_diff(a, b):
    raise NotImplementedError


def mean_abs_diff(a, b):
    raise NotImplementedError
