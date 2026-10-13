"""Classify a cell: X, 0, or empty

Course: exam Ex2 (X and 0, both sessions in 2025). This is the decision you
make per cell after cutting up the board. Here the cells are synthetic so you
can focus on the rule; the mock exams run it on the real photos.

TASK
  classify_cell(cell) -> "X", "0" or "empty"
      cell: grayscale uint8, light paper, dark ink, roughly square. Strokes
      vary in thickness and position; some cells have a dark grid line along
      the top edge (left over from cropping).

ANY cv2 function is allowed. Target: >= 97% on 90 random cells.

HINTS  hidden on purpose: python check.py d33 --hint (one step at a time)
"""


def classify_cell(cell):
    raise NotImplementedError
