"""Non-maximum suppression along the gradient (thin edges)

Course: Curs 4 (Canny). Gradient magnitude gives THICK edges: a blurred edge is
several pixels wide. Canny keeps only pixels that are a maximum along the
gradient direction, giving 1-pixel-wide edges. This is the step people skip
when they "use Canny" without understanding it.

TASK
  nonmax_suppression(mag, ang) -> float64 HxW
      mag, ang come from d12 (ang in degrees, [0, 360), y grows down).
      Quantise the direction (mod 180) into 4 bins:
          [0, 22.5) or [157.5, 180) -> compare with left/right neighbours
          [22.5, 67.5)              -> compare with (y+1, x+1) and (y-1, x-1)
          [67.5, 112.5)             -> compare with up/down neighbours
          [112.5, 157.5)            -> compare with (y+1, x-1) and (y-1, x+1)
      Keep mag[y, x] if it is >= both neighbours, else 0. Set the 1-pixel image
      border to 0.

NOTE   A 45 degree edge ends up 2 pixels wide per row with this scheme, because
       diagonal neighbours skip every other pixel. Real Canny interpolates the
       magnitude between neighbours to fix that. The test allows it.

HINTS  hidden on purpose: python check.py d20 --hint (one step at a time)
"""


def nonmax_suppression(mag, ang):
    raise NotImplementedError
