"""Intersection over union, the evaluator's way

Course: Lab 9-10, Tema 2. A detection counts as correct when its IoU with a
ground-truth face is >= 0.3 (Tema 2 evaluator). The official code treats
coordinates as INCLUSIVE pixels: a box (0, 0, 9, 9) is 10 x 10 = 100 pixels.

TASK
  iou(a, b) -> float, boxes (xmin, ymin, xmax, ymax), inclusive (+1 on widths).
      Non-overlapping boxes give 0.0.
  iou_matrix(A, B) -> (len(A), len(B)) array of IoUs, vectorised (no Python
      loop over pairs).
"""


def iou(a, b):
    raise NotImplementedError


def iou_matrix(A, B):
    raise NotImplementedError
