"""Count the pips on a dice face

Course: exam Ex2, June 2025 (3 dice squares, output the sum). Counting blobs
is a component-labelling problem plus filters that reject everything that
isn't a dot: the square's border, noise specks, glare.

TASK
  count_pips(bgr) -> int in 1..6
      bgr: one dice square (light background, dark round pips, a dark border
      line around the square, noise). Target: 100% on 60 squares.

ANY cv2 function is allowed.

IDEAS
  - Crop a margin to drop the border, threshold (Otsu, inverted), open to
    kill specks, then cv2.connectedComponentsWithStats and keep components
    whose area is in a plausible range.
  - Or cv2.HoughCircles with a radius range tied to the square size.
  - Or findContours + circularity filter (4*pi*A/P^2 close to 1).
  Make every size threshold RELATIVE to the square size: test squares vary.
"""


def count_pips(bgr):
    raise NotImplementedError
