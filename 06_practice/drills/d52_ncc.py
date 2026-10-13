"""Template matching with normalised cross-correlation

Course: Curs 3 (finding templates with filters), Qwirkle shape templates,
exam Ex2 (car pictograms, digits). Plain correlation prefers bright regions;
normalising by mean and energy makes the score a correlation coefficient in
[-1, 1] that ignores brightness and contrast.

TASK
  match_ncc(img, tpl) -> float64 map of shape (H - h + 1, W - w + 1)
      Same as cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED):
      at each position, with patch P and template T (both mean-subtracted),
      score = sum(P*T) / sqrt(sum(P^2) * sum(T^2)). Define the score as 0
      where the patch is perfectly flat.
  best_match(img, tpl) -> (x, y) top-left corner of the best match.

RULES  no cv2.matchTemplate.
HINTS  hidden on purpose: python check.py d52 --hint (one step at a time)
"""


def match_ncc(img, tpl):
    raise NotImplementedError


def best_match(img, tpl):
    raise NotImplementedError
