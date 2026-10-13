"""Count the pips on a dice face

Course: exam Ex2, June 2025 (3 dice squares, output the sum). Counting blobs
is a component-labelling problem plus filters that reject everything that
isn't a dot: the square's border, noise specks, glare.

TASK
  count_pips(bgr) -> int in 1..6
      bgr: one dice square (light background, dark round pips, a dark border
      line around the square, noise). Target: 100% on 60 squares.

ANY cv2 function is allowed.

HINTS  hidden on purpose: python check.py d53 --hint (one step at a time)
"""


def count_pips(bgr):
    raise NotImplementedError
