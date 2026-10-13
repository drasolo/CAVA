"""RANSAC line fitting

Course: Curs 5. Least squares is ruined by outliers; RANSAC samples minimal
sets, keeps the model most points agree with, and ignores the rest. Also know
the iteration formula; it's a classic written-exam question.

TASK
  ransac_iterations(p, r, k) -> int
      Smallest N with 1 - (1 - r^k)^N >= p   (p: success probability,
      r: inlier ratio, k: points per sample).
  fit_line_ransac(points, thresh, n_iter, seed=0) -> ((a, b, c), inlier_mask)
      points: Nx2 array of (x, y). Line a*x + b*y + c = 0 with a^2 + b^2 = 1.
      Each iteration: pick 2 distinct random points, build the line through
      them, count points with |a x + b y + c| < thresh. Keep the best.
      Finally refit on the best inliers with total least squares (SVD /
      principal direction) and return that line and the inlier mask.

RULES  no sklearn, cv2.fitLine.
HINT   Line through p1, p2: direction d = p2 - p1, normal n = (-d_y, d_x)/|d|,
       c = -n . p1. For the refit: centre the inliers; the normal is the
       singular vector with the SMALLEST singular value.
"""


def ransac_iterations(p, r, k):
    raise NotImplementedError


def fit_line_ransac(points, thresh, n_iter, seed=0):
    raise NotImplementedError
