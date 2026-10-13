import numpy as np


def _iou(a, b):
    xa, ya, xb, yb = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    area = lambda r: (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
    return inter / float(area(a) + area(b) - inter)


def average_precision(detections, scores, file_names, gt):
    order = np.argsort(np.asarray(scores))[::-1]
    dets = np.asarray(detections)[order]
    names = np.asarray(file_names)[order]
    gt_names = np.array([g[0] for g in gt])
    gt_boxes = [g[1] for g in gt]
    used = np.zeros(len(gt), bool)
    tp = np.zeros(len(dets))
    fp = np.zeros(len(dets))
    for k, (box, name) in enumerate(zip(dets, names)):
        idx = np.nonzero(gt_names == name)[0]
        best, best_i = -1.0, -1
        for i in idx:
            o = _iou(box, gt_boxes[i])
            if o > best:
                best, best_i = o, i
        if best >= 0.3 and not used[best_i]:
            tp[k] = 1
            used[best_i] = True
        else:
            fp[k] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / len(gt)
    prec = ctp / (ctp + cfp)
    m_rec = np.concatenate(([0], rec, [1]))
    m_pre = np.concatenate(([0], prec, [0]))
    i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
    return float(np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i]))
