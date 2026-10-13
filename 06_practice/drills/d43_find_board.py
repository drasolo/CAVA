"""Find the board and warp it flat

Course: Lab 1/6 (Sudoku), exam Ex2, Qwirkle. This is the step that fails
first on every board problem. Classic recipe: grayscale -> blur -> threshold
(or Canny) -> biggest contour -> approxPolyDP to 4 points -> order -> warp.

TASK
  find_board(bgr) -> 4x2 float32 corners TL, TR, BR, BL of the light board.
      Scenes: a light grid board, slightly rotated and in perspective, on a
      dark table full of clutter (blobs, lines). Target: every corner within
      6 px on 20 scenes.
  warp_board(bgr, corners, side) -> side x side x 3 straightened board.

ANY cv2 function is allowed (findContours, approxPolyDP, getPerspectiveTransform,
warpPerspective...). Reuse your d40 idea for ordering.

HINTS  hidden on purpose: python check.py d43 --hint (one step at a time)
"""


def find_board(bgr):
    raise NotImplementedError


def warp_board(bgr, corners, side):
    raise NotImplementedError
