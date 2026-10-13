import numpy as np
from cava_practice.testing import check, close


def official_ap(detections, scores, file_names, gt):
    """Transcribed from 04_tema2-scooby/evaluare/cod_evaluare/evalueaza_solutie.py."""
    gt_names = np.array([g[0] for g in gt])
    gt_dets = np.array([g[1] for g in gt])
    exists = np.zeros(len(gt))
    idx = np.argsort(scores)[::-1]
    file_names, detections = file_names[idx], detections[idx]
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    for d in range(len(detections)):
        on_img = np.where(gt_names == file_names[d])[0]
        best, best_i = -1, -1
        for gi, g in enumerate(gt_dets[on_img]):
            b = detections[d]
            xa, ya, xb, yb = max(b[0], g[0]), max(b[1], g[1]), min(b[2], g[2]), min(b[3], g[3])
            inter = max(0, xb - xa + 1) * max(0, yb - ya + 1)
            o = inter / float((b[2] - b[0] + 1) * (b[3] - b[1] + 1) + (g[2] - g[0] + 1) * (g[3] - g[1] + 1) - inter)
            if o > best:
                best, best_i = o, on_img[gi]
        if best >= 0.3 and exists[best_i] == 0:
            tp[d] = 1
            exists[best_i] = 1
        else:
            fp[d] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec, prec = ctp / len(gt), ctp / (ctp + cfp)
    m_rec = np.concatenate(([0], rec, [1]))
    m_pre = np.concatenate(([0], prec, [0]))
    for i in range(len(m_pre) - 1, -1, 1):  # never runs: kept as in the original
        m_pre[i] = max(m_pre[i], m_pre[i + 1])
    i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
    return np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])


def scenario(seed):
    r = np.random.default_rng(seed)
    gt, dets, names = [], [], []
    for k in range(12):
        name = f"{k:04d}.jpg"
        for _ in range(int(r.integers(0, 4))):
            x, y = r.integers(0, 300, 2)
            s = int(r.integers(30, 90))
            g = (int(x), int(y), int(x + s), int(y + s))
            gt.append((name, g))
            for _ in range(int(r.integers(0, 3))):  # hits and duplicates
                j = r.integers(-12, 13, 4)
                dets.append(tuple(int(v) for v in np.array(g) + j))
                names.append(name)
        for _ in range(int(r.integers(0, 4))):  # false alarms
            x, y = r.integers(0, 300, 2)
            dets.append((int(x), int(y), int(x + 40), int(y + 40)))
            names.append(name)
    return np.array(dets), r.random(len(dets)), np.array(names), gt


def test(m):
    dets = np.array([(0, 0, 9, 9)])
    close(m.average_precision(dets, np.array([1.0]), np.array(["a.jpg"]), [("a.jpg", (0, 0, 9, 9))]), 1.0, 1e-12, "one perfect detection")
    for seed in range(8):
        d, s, n, gt = scenario(seed)
        if not gt or not len(d):
            continue
        close(m.average_precision(d, s, n, gt), official_ap(d, s, n, gt), 1e-9, f"random scenario {seed}")
    d, s, n, gt = scenario(3)
    ap_good = m.average_precision(d, s, n, gt)
    check(0 < ap_good < 1, "sanity: scenario AP should be strictly between 0 and 1")
