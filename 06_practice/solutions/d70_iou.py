import numpy as np


def iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / float(area_a + area_b - inter)


def iou_matrix(A, B):
    A = np.asarray(A, np.float64)[:, None, :]
    B = np.asarray(B, np.float64)[None, :, :]
    w = np.clip(np.minimum(A[..., 2], B[..., 2]) - np.maximum(A[..., 0], B[..., 0]) + 1, 0, None)
    h = np.clip(np.minimum(A[..., 3], B[..., 3]) - np.maximum(A[..., 1], B[..., 1]) + 1, 0, None)
    inter = w * h
    area = lambda X: (X[..., 2] - X[..., 0] + 1) * (X[..., 3] - X[..., 1] + 1)
    return inter / (area(A) + area(B) - inter)
