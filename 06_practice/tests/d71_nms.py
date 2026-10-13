import numpy as np
from cava_practice.testing import check


def test(m):
    boxes = [(0, 0, 9, 9), (1, 1, 10, 10), (50, 50, 59, 59), (2, 0, 11, 9), (51, 50, 60, 59)]
    scores = [0.9, 0.95, 0.5, 0.3, 0.7]
    got = list(m.nms(boxes, scores, 0.3))
    check(got == [1, 4], f"expected [1, 4] (best of each cluster, highest score first), got {got}")
    check(list(m.nms(boxes, scores, 0.99)) == [1, 0, 4, 2, 3], "with a threshold of 0.99 nothing overlapping enough: all kept, sorted by score")
    boxes = [(0, 0, 9, 9), (0, 0, 9, 9), (0, 0, 9, 9)]
    ids = ["a.jpg", "b.jpg", "a.jpg"]
    got = list(m.nms_per_image(boxes, [0.5, 0.4, 0.6], ids, 0.3))
    check(got == [1, 2], f"identical boxes in DIFFERENT images must not suppress each other: expected [1, 2], got {got}")
    r = np.random.default_rng(3)
    xy = r.integers(0, 200, (60, 2))
    wh = r.integers(10, 40, (60, 2))
    B = np.c_[xy, xy + wh]
    S = r.random(60)
    keep = list(m.nms(B, S, 0.3))
    check(len(set(keep)) == len(keep), "kept indices must be unique")
    check(all(S[keep[i]] >= S[keep[i + 1]] for i in range(len(keep) - 1)), "kept indices must be sorted by score, highest first")
