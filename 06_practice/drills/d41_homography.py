"""Perspective transform (homography) from 4 points

Course: Curs 5 (panoramas), every board exercise (rectify the board). A
homography H maps (x, y, 1) -> (u', v', w') and the point is (u'/w', v'/w').
Four point pairs give 8 equations for the 8 unknowns (h33 = 1).

TASK
  homography(src, dst) -> 3x3 float64 H with H[2, 2] == 1, mapping src -> dst.
      src, dst: 4x2 arrays of (x, y).
  apply_h(H, pts) -> Nx2 array, H applied to each point (with the divide).

RULES  no cv2.getPerspectiveTransform, cv2.findHomography, cv2.perspectiveTransform.
HINTS  hidden on purpose: python check.py d41 --hint (one step at a time)
"""


def homography(src, dst):
    raise NotImplementedError


def apply_h(H, pts):
    raise NotImplementedError
