import numpy as np


def _iou_one(b, B):
    w = np.clip(np.minimum(b[2], B[:, 2]) - np.maximum(b[0], B[:, 0]) + 1, 0, None)
    h = np.clip(np.minimum(b[3], B[:, 3]) - np.maximum(b[1], B[:, 1]) + 1, 0, None)
    inter = w * h
    area = lambda X: (X[..., 2] - X[..., 0] + 1) * (X[..., 3] - X[..., 1] + 1)
    return inter / (area(b) + area(B) - inter)


def nms(boxes, scores, iou_thr):
    boxes = np.asarray(boxes, np.float64)
    order = list(np.argsort(-np.asarray(scores), kind="stable"))
    keep = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        if order:
            ious = _iou_one(boxes[i], boxes[order])
            order = [j for j, v in zip(order, ious) if v <= iou_thr]
    return keep


def nms_per_image(boxes, scores, image_ids, iou_thr):
    boxes, scores, ids = np.asarray(boxes), np.asarray(scores), np.asarray(image_ids)
    keep = []
    for img in np.unique(ids):
        idx = np.nonzero(ids == img)[0]
        keep += [int(idx[k]) for k in nms(boxes[idx], scores[idx], iou_thr)]
    return sorted(keep)
