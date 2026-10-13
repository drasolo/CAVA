"""Perspective transform (homography) from 4 points

Course: Curs 5 (panoramas), every board exercise (rectify the board). A
homography H maps (x, y, 1) -> (u', v', w') and the point is (u'/w', v'/w').
Four point pairs give 8 equations for the 8 unknowns (h33 = 1).

TASK
  homography(src, dst) -> 3x3 float64 H with H[2, 2] == 1, mapping src -> dst.
      src, dst: 4x2 arrays of (x, y).
  apply_h(H, pts) -> Nx2 array, H applied to each point (with the divide).

RULES  no cv2.getPerspectiveTransform, cv2.findHomography, cv2.perspectiveTransform.
HINT   For each pair (x, y) -> (u, v), two rows of A h = b:
         [x, y, 1, 0, 0, 0, -u*x, -u*y] . h = u
         [0, 0, 0, x, y, 1, -v*x, -v*y] . h = v
       Solve with np.linalg.solve, append 1, reshape to 3x3.
"""


def homography(src, dst):
    raise NotImplementedError


def apply_h(H, pts):
    raise NotImplementedError
