"""Correlation and convolution from scratch

Course: Curs 2-3 (linear filters). Every blur, every gradient, and template
matching is this one operation. Know the difference: correlation slides the
kernel as-is; convolution flips it first (both axes). cv2.filter2D does
CORRELATION.

TASK
  correlate2d(img, kernel) -> float64 array, same HxW as img
      out[y, x] = sum_{i,j} kernel[i, j] * img[y + i - ky, x + j - kx]
      where (ky, kx) is the kernel centre. Kernels have odd sizes but need not
      be square (e.g. 3x5). Pad with BORDER_REPLICATE (repeat the edge pixel).
  convolve2d(img, kernel) -> same, but with the kernel flipped.

RULES  no cv2.filter2D, cv2.sepFilter2D, scipy.
HINT   np.pad(img, ..., mode="edge") then loop over the KERNEL entries (a few),
       not over the pixels (many): add kernel[i, j] * shifted_slice.
"""


def correlate2d(img, kernel):
    raise NotImplementedError


def convolve2d(img, kernel):
    raise NotImplementedError
