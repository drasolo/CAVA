"""Order 4 corners: TL, TR, BR, BL

Course: every board exercise. approxPolyDP returns the 4 corners of the board
in an arbitrary order and starting point. getPerspectiveTransform needs them in
the SAME order as your destination points, or the board comes out rotated or
mirrored (and every cell index is wrong).

TASK
  order_corners(pts) -> float32 array 4x2 in order top-left, top-right,
      bottom-right, bottom-left. pts: any 4x2 / 4x1x2 array of (x, y), any order.
      Must work for boards rotated by up to +-30 degrees.

HINT   Classic trick: TL has the smallest x+y, BR the largest; TR has the
       smallest y-x, BL the largest. Fine up to ~30 degrees, but near 45
       degrees two corners tie and it picks the same point twice. Sturdier:
       sort by angle around the centroid (a consistent cycle), then rotate the
       cycle to start at the smallest x+y. Try both.
"""


def order_corners(pts):
    raise NotImplementedError
