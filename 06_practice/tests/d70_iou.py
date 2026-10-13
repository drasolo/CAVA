import numpy as np
from cava_practice.testing import check, close


def official(a, b):
    x_a, y_a = max(a[0], b[0]), max(a[1], b[1])
    x_b, y_b = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    return inter / float((a[2] - a[0] + 1) * (a[3] - a[1] + 1) + (b[2] - b[0] + 1) * (b[3] - b[1] + 1) - inter)


def test(m):
    close(m.iou((0, 0, 9, 9), (0, 0, 9, 9)), 1.0, 1e-12, "identical boxes")
    close(m.iou((0, 0, 9, 9), (5, 0, 14, 9)), 50 / 150, 1e-12, "half overlap (inclusive pixels: 50 / 150)")
    close(m.iou((0, 0, 9, 9), (20, 20, 30, 30)), 0.0, 1e-12, "disjoint boxes")
    close(m.iou((0, 0, 9, 9), (10, 0, 19, 9)), 0.0, 1e-12, "boxes touching edge to edge share no pixel")
    r = np.random.default_rng(0)
    A = np.sort(r.integers(0, 100, (15, 2, 2)), axis=1).transpose(0, 2, 1).reshape(15, 4)[:, [0, 2, 1, 3]]
    B = np.sort(r.integers(0, 100, (11, 2, 2)), axis=1).transpose(0, 2, 1).reshape(11, 4)[:, [0, 2, 1, 3]]
    want = np.array([[official(a, b) for b in B] for a in A])
    for i in range(3):
        close(m.iou(A[i], B[i]), want[i, i], 1e-12, f"random pair {i}")
    got = np.asarray(m.iou_matrix(A, B))
    check(got.shape == (15, 11), f"iou_matrix shape {got.shape}, expected (15, 11)")
    close(got, want, 1e-12, "iou_matrix")
