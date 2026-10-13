"""Hough transform for lines, from scratch

Course: Curs 5. Polar form x*cos(t) + y*sin(t) = rho, so vertical lines are
fine (y = mx + b can't represent them). Every edge pixel votes for all lines
through it; peaks in the accumulator are lines. This is how you'd find board
grid lines when contours fail.

TASK
  hough_accumulator(edges, theta_step=1.0, rho_step=1.0) -> (acc, rhos, thetas)
      edges: HxW, nonzero = edge pixel.
      thetas: degrees, np.arange(0, 180, theta_step).
      rhos:   np.arange(-D, D + rho_step, rho_step), D = ceil(hypot(H, W)).
      acc[i, j] = number of edge pixels on the line (rhos[i], thetas[j]),
      rounding each pixel's rho to the nearest bin.
  top_lines(acc, rhos, thetas, k, nms=5) -> list of k (rho, theta) tuples
      Strongest first. After taking a peak, zero out the acc window within
      +-nms bins around it so one thick line isn't reported twice.

RULES  no cv2.HoughLines, cv2.HoughLinesP, skimage.transform.
HINT   Vectorise over theta: for all edge points (ys, xs) and all thetas,
       rho = xs[:, None]*cos + ys[:, None]*sin; then np.add.at(acc, (ri, ti), 1).
"""


def hough_accumulator(edges, theta_step=1.0, rho_step=1.0):
    raise NotImplementedError


def top_lines(acc, rhos, thetas, k, nms=5):
    raise NotImplementedError
