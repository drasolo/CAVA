from __future__ import annotations
import numpy as np

def verify_feature_variability(pos_feat: np.ndarray, neg_feat: np.ndarray) -> dict:
    """
    Returns stats + prints [OK] lines.
    Required prints:
      - ||pos0-pos1||, ||neg0-neg1||, ||pos0-neg0||
      - POS/NEG mean std per-dim
    """
    if len(pos_feat) < 3 or len(neg_feat) < 3:
        raise ValueError("Need at least 3 pos and 3 neg samples for variability checks")

    d_pos = float(np.linalg.norm(pos_feat[0] - pos_feat[1]))
    d_neg = float(np.linalg.norm(neg_feat[0] - neg_feat[1]))
    d_cross = float(np.linalg.norm(pos_feat[0] - neg_feat[0]))

    pos_std = float(pos_feat.std(axis=0).mean())
    neg_std = float(neg_feat.std(axis=0).mean())

    print(f"[OK] feat var: ||pos0-pos1||={d_pos:.4f}  ||neg0-neg1||={d_neg:.4f}  ||pos0-neg0||={d_cross:.4f}")
    print(f"[OK] feat var: POS mean std/dim={pos_std:.6f}  NEG mean std/dim={neg_std:.6f}")

    return {
        "d_pos": d_pos,
        "d_neg": d_neg,
        "d_cross": d_cross,
        "pos_std_mean": pos_std,
        "neg_std_mean": neg_std,
    }


def iou_xyxy(a, b) -> float:
    """
    Intersection-over-Union for boxes in [x1, y1, x2, y2] format (inclusive coords).
    Returns IoU in [0, 1].
    """
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1 + 1.0)
    inter_h = max(0.0, inter_y2 - inter_y1 + 1.0)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1 + 1.0) * max(0.0, ay2 - ay1 + 1.0)
    area_b = max(0.0, bx2 - bx1 + 1.0) * max(0.0, by2 - by1 + 1.0)

    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0

    return inter_area / denom


def _toy_tests() -> None:
    # identical boxes -> IoU = 1
    a = [10, 10, 20, 20]
    assert abs(iou_xyxy(a, a) - 1.0) < 1e-9

    # disjoint boxes -> IoU = 0
    b = [100, 100, 110, 110]
    assert iou_xyxy(a, b) == 0.0

    # partial overlap sanity (should be between 0 and 1)
    c = [15, 15, 25, 25]
    v = iou_xyxy(a, c)
    assert 0.0 < v < 1.0

    print("[OK] metrics.py toy tests passed")



def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """
    Greedy Non-Maximum Suppression for boxes in [x1, y1, x2, y2] (inclusive coords).

    Returns indices to KEEP (indices refer to the input arrays).
    The function sorts internally by score descending.
    """
    if boxes is None or len(boxes) == 0:
        return []

    if scores is None or len(scores) == 0:
        return []

    if len(boxes) != len(scores):
        raise ValueError(f"boxes and scores must have same length: {len(boxes)} vs {len(scores)}")

    boxes = boxes.astype(np.float32, copy=False)
    scores = scores.astype(np.float32, copy=False)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)

    order = np.argsort(scores)[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h

        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)

        order = rest[iou < iou_thr]

    return keep




if __name__ == "__main__":
    _toy_tests()
