import cv2
import numpy as np
from cava_practice import synth
from cava_practice.testing import check, close, forbid


def test(m):
    forbid(m, ["matchTemplate"])
    img = synth.smooth_image(60, 80, seed=8)
    for (y, x, h, w) in [(10, 20, 12, 12), (30, 5, 9, 15), (0, 0, 20, 7)]:
        tpl = img[y:y + h, x:x + w].copy()
        ref = cv2.matchTemplate(img.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        got = np.asarray(m.match_ncc(img, tpl))
        check(got.shape == ref.shape, f"template {h}x{w}: map shape {got.shape}, expected {ref.shape} = (H-h+1, W-w+1)")
        close(got, ref, 1e-3, f"template {h}x{w}: NCC map vs cv2")
        check(tuple(m.best_match(img, tpl)) == (x, y), f"best_match should return (x, y) = {(x, y)}, got {m.best_match(img, tpl)}")
    # brightness/contrast invariance: a darker, lower-contrast copy must still match perfectly
    tpl = img[25:40, 40:60].astype(np.float64) * 0.5 + 30
    check(tuple(m.best_match(img, tpl)) == (40, 25), "NCC must ignore brightness and contrast changes")
