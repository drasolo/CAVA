"""Cut the board into cells and number them the way the statement says

Course: every grid exercise. Output is graded all-or-nothing per image, so one
numbering mistake zeroes a perfectly recognised board. Seen so far:
  - X&0 (2025):        3x3, numbered 1..9 row by row ("top to bottom, left to right")
  - Math ring (2026):  16 squares on the BORDER of a 5x5 grid, numbered clockwise
                       starting at the top-left corner (1-5 top row left->right,
                       6-9 right column downward, 10-13 bottom row right->left,
                       14-16 left column upward). The middle is empty paper.
  - Qwirkle (Tema 1):  16x16, positions like "2H" = row 2, column H (A..P).

TASK
  cell_boxes(side, n, margin=0.0) -> list of n*n (x0, y0, x1, y1) int tuples,
      row by row, for a warped side x side board. Cell k spans
      [k*side/n, (k+1)*side/n). margin is a fraction of the cell size removed
      from EACH edge (0.1 -> drop 10% on every side) to cut off grid lines.
      x1, y1 are exclusive (use them as img[y0:y1, x0:x1]).
  number_map(n, scheme) -> n x n int array
      scheme "row_major": 1..n*n row by row.
      scheme "ring_cw":   border cells numbered clockwise from the top-left
                          corner starting at 1; interior cells are 0.
  qwirkle_pos(row, col) -> str   0-based (row, col) -> "2H"-style label.
  parse_qwirkle_pos(s)  -> (row, col) 0-based. Inverse of the above, and must
      handle two-digit rows like "16P".
"""


def cell_boxes(side, n, margin=0.0):
    raise NotImplementedError


def number_map(n, scheme):
    raise NotImplementedError


def qwirkle_pos(row, col):
    raise NotImplementedError


def parse_qwirkle_pos(s):
    raise NotImplementedError
