"""Read annotations and crop boxes safely

Course: exam Ex1 (with bounding boxes), Tema 2. Every face exercise starts by
parsing "name xmin ymin xmax ymax [character]" and cutting the face out.

TASK
  parse_annotations(lines) -> list of dicts
      Input: lines of an annotation file (strings, may have trailing spaces or
      blank lines). Each output dict: {"file": str, "box": (xmin, ymin, xmax, ymax)
      as ints, "label": str or None}.
  crop(img, box) -> sub-image
      Coordinates are INCLUSIVE pixel indices (the official evaluator adds +1
      to widths), so a box (2, 3, 5, 9) is 4 pixels wide and 7 tall.
      Clip the box to the image: boxes may stick out past the border or be
      negative.

HINTS  hidden on purpose: python check.py d01 --hint (one step at a time)
"""


def parse_annotations(lines):
    raise NotImplementedError


def crop(img, box):
    raise NotImplementedError
