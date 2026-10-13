"""Erosion, dilation, opening, closing

Course: Curs 6. After thresholding you always have specks and holes.
Opening (erode then dilate) removes small white specks; closing (dilate then
erode) fills small black holes and gaps. Know which is which without thinking.

TASK
  erode(binary, k), dilate(binary, k)  -> uint8 0/255, square k x k element.
      Pixels outside the image must not affect the result (pad erosion with
      255 and dilation with 0).
  opening(binary, k), closing(binary, k)
  ANSWERS = {"remove_white_specks": ..., "fill_black_holes": ...}
      each "opening" or "closing".

RULES  no cv2.erode, cv2.dilate, cv2.morphologyEx, scipy, skimage.
HINT   Erosion = min over the window, dilation = max. Reuse the shifted-slice
       trick from d10 with np.minimum / np.maximum.
"""

ANSWERS = {"remove_white_specks": None, "fill_black_holes": None}


def erode(binary, k):
    raise NotImplementedError


def dilate(binary, k):
    raise NotImplementedError


def opening(binary, k):
    raise NotImplementedError


def closing(binary, k):
    raise NotImplementedError
