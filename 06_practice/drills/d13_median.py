"""Median filter, and choosing the right denoiser

Course: Curs 2-3. Salt-and-pepper noise (isolated black/white pixels) survives
a mean or Gaussian blur as grey smudges; a median removes it outright and
keeps edges sharp. Gaussian noise is the opposite case.

TASK
  median_filter(gray, k) -> uint8 HxW, k odd. Each pixel becomes the median of
      its k x k neighbourhood. Pad with replicated edges.
  CHOICE  dict saying which filter to use for each noise type:
      {"salt_pepper": "median" or "gaussian", "gaussian_noise": "median" or "gaussian"}

RULES  no cv2.medianBlur, scipy.
HINTS  hidden on purpose: python check.py d13 --hint (one step at a time)
"""

CHOICE = {"salt_pepper": None, "gaussian_noise": None}


def median_filter(gray, k):
    raise NotImplementedError
