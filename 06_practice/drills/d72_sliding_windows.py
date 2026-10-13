"""Sliding windows over an image pyramid

Course: Lab 9-10 (face detection), Tema 2 (the base solution MUST be sliding
window + features + classifier). The classifier only sees fixed-size windows
(say 64x64). To find faces of other sizes you shrink or enlarge the IMAGE,
not the window, then map each window back to original coordinates.

TASK
  window_boxes(img_h, img_w, win, step, scales) -> int array (N, 4)
      For each scale s in `scales` (in order): the image is resized to
      (round(img_h * s), round(img_w * s)). Windows of win x win start at
      x = 0, step, 2*step, ... while x + win <= resized width (same for y,
      rows in the outer loop). Map each window back to the original image:
          (floor(x / s), floor(y / s), floor((x + win) / s) - 1, floor((y + win) / s) - 1)
      (inclusive coordinates, as the evaluator expects).
  find_square(img, template, scales, step) -> (xmin, ymin, xmax, ymax)
      The image contains one copy of `template`, scaled by an unknown factor.
      Build the pyramid with cv2.resize, score every window with normalised
      cross-correlation against the template (cv2.matchTemplate is allowed
      here as the "classifier"), and return the best window in ORIGINAL
      coordinates.
"""


def window_boxes(img_h, img_w, win, step, scales):
    raise NotImplementedError


def find_square(img, template, scales, step):
    raise NotImplementedError
