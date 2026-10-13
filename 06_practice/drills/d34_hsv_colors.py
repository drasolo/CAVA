"""Colour by HSV (and the red hue wrap-around)

Course: Curs 6-7 (RGB vs HSV), Lab 7 (colour filtering), Qwirkle (6 tile
colours). In OpenCV, H runs 0..179 (degrees / 2), S and V 0..255. Red sits at
BOTH ends of the hue range, so a single inRange misses half of it.

TASK
  red_mask(bgr) -> uint8 0/255 mask of red pixels (any H near 0 OR near 179,
      with enough saturation and value).
  color_name(bgr_patch) -> one of "red", "orange", "yellow", "green", "blue",
      "white" for a small patch that is mostly one colour (with noise and
      brightness changes).

ANY cv2 function is allowed. color_name target: >= 97% on 180 patches.

HINTS  hidden on purpose: python check.py d34 --hint (one step at a time)
"""


def red_mask(bgr):
    raise NotImplementedError


def color_name(bgr_patch):
    raise NotImplementedError
