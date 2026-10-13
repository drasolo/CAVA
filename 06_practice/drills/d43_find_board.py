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

HINTS  The board is the largest bright region, BUT light clutter can touch
       it and a plain brightness threshold merges the two into one contour.
       Threshold on what makes the board unique (bright AND unsaturated, in
       HSV), then clean the mask with morphology. approxPolyDP's epsilon is a
       fraction of the perimeter (start at 0.02 * cv2.arcLength(c, True)). If
       you don't get exactly 4 points, adjust epsilon in a loop, or fall back
       to cv2.minAreaRect / the convex hull.
       Debug by drawing your corners on the image and saving it.
"""


def find_board(bgr):
    raise NotImplementedError


def warp_board(bgr, corners, side):
    raise NotImplementedError
