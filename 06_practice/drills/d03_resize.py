"""Resizing, and the (width, height) trap

Course: HOG needs every window at one fixed size; exam Ex1 faces come in all
sizes. cv2.resize takes dsize=(WIDTH, HEIGHT), but img.shape is (HEIGHT, WIDTH).

TASK
  resize_nearest(img, dsize) -> image of shape (dsize[1], dsize[0], ...)
      Nearest-neighbour: output pixel (y, x) copies source pixel
      (floor(y * H_src / H_dst), floor(x * W_src / W_dst)).
  fit_square(img, side) -> side x side image
      Resize keeping the aspect ratio so the LONGER side equals `side`, then
      pad the short side symmetrically with zeros (letterboxing). cv2.resize
      with INTER_AREA is allowed here.

RULES  resize_nearest must not use cv2.resize.
HINT   Build index arrays with np.arange and use fancy indexing img[ys][:, xs].
"""


def resize_nearest(img, dsize):
    raise NotImplementedError


def fit_square(img, side):
    raise NotImplementedError
