"""Image gradients with Sobel

Course: Curs 4 (gradients, edges). The gradient is the input to Canny, Hough
(with the gradient extension), and HOG. Get its direction convention right
once and for all: x grows to the right, y grows DOWN.

TASK
  sobel(gray) -> (gx, gy), both float64 HxW
      gx responds to intensity increasing to the right, gy to intensity
      increasing downward. Use the 3x3 Sobel kernels with cv2.filter2D
      (allowed here; default border).
  magnitude_orientation(gx, gy) -> (mag, ang)
      mag = sqrt(gx^2 + gy^2); ang in DEGREES in [0, 360).

RULES  no cv2.Sobel, cv2.Scharr, cv2.spatialGradient, cv2.cartToPolar, cv2.phase.
HINT   np.arctan2(gy, gx) returns radians in (-pi, pi]; shift negatives up by 360.
"""


def sobel(gray):
    raise NotImplementedError


def magnitude_orientation(gx, gy):
    raise NotImplementedError
