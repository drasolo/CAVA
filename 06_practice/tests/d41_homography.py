import cv2
import numpy as np
from cava_practice import synth
from cava_practice.testing import check, close, forbid


def test(m):
    forbid(m, ["getPerspectiveTransform", "findHomography", "perspectiveTransform"])
    for seed in range(5):
        _, corners = synth.board_scene(seed)
        S = 450
        dst = np.float32([[0, 0], [S - 1, 0], [S - 1, S - 1], [0, S - 1]])
        H = np.asarray(m.homography(corners, dst), np.float64)
        ref = cv2.getPerspectiveTransform(corners, dst)
        check(H.shape == (3, 3) and abs(H[2, 2] - 1) < 1e-9, "H must be 3x3 with H[2, 2] == 1")
        close(H, ref, 1e-6 * np.abs(ref).max() + 1e-9, f"seed {seed}: homography vs OpenCV")
        pts = np.random.default_rng(seed).uniform(0, 600, (10, 2))
        want = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), ref).reshape(-1, 2)
        close(m.apply_h(H, pts), want, 1e-4, f"seed {seed}: apply_h (did you divide by w?)")
        close(m.apply_h(H, corners), dst, 1e-3, f"seed {seed}: corners must land on the destination square")
