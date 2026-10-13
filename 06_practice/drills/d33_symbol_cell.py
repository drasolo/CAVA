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

IDEAS (pick one, then try another)
  1. Drop a margin (~10-15%) to remove grid lines. Threshold (Otsu, inverted
     so ink = 255). Empty if there is almost no ink, OR if the contrast between
     the two Otsu classes is small (Otsu ALWAYS splits, even pure noise).
  2. 0 has a HOLE, X doesn't: cv2.findContours with RETR_CCOMP and look for a
     child contour of reasonable area.
  3. Or: the centre of a 0 is paper, the centre of an X is ink.
  4. Or: circularity 4*pi*area/perimeter^2 of the largest contour.
"""


def classify_cell(cell):
    raise NotImplementedError
