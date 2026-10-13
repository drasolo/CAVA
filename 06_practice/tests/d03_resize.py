import cv2
import numpy as np
from cava_practice import synth
from cava_practice.testing import check, forbid


def test(m):
    forbid(m.resize_nearest, ["cv2.resize", "resize"])
    img = synth.random_image(60, 90, seed=3)
    out = m.resize_nearest(img, (45, 20))
    check(out.shape == (20, 45, 3), f"dsize=(45, 20) means width 45, height 20 -> shape (20, 45, 3); got {out.shape}")
    for dsize in [(45, 20), (128, 64), (13, 97)]:
        ref = cv2.resize(img, dsize, interpolation=cv2.INTER_NEAREST)
        got = m.resize_nearest(img, dsize)
        check(got.shape == ref.shape, f"dsize {dsize}: shape {got.shape}, expected {ref.shape}")
        agree = (got == ref).all(axis=-1).mean()
        check(agree > 0.99, f"dsize {dsize}: only {agree:.1%} of pixels match cv2 INTER_NEAREST")
    tall = np.full((100, 40, 3), 200, np.uint8)
    sq = m.fit_square(tall, 64)
    check(sq.shape == (64, 64, 3), f"fit_square shape {sq.shape}, expected (64, 64, 3)")
    check(sq[32, 0].sum() == 0 and sq[32, 63].sum() == 0, "a tall image must get black padding left and right")
    check(sq[0, 32].sum() > 0 and sq[63, 32].sum() > 0, "a tall image must fill the full height")
    cols = np.where(sq[32, :, 0] > 0)[0]
    check(abs(len(cols) - 26) <= 1, f"content width should be ~26 px (40*64/100), got {len(cols)}")
    check(abs(cols[0] - (63 - cols[-1])) <= 1, "padding must be symmetric")
