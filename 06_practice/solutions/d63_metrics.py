import numpy as np


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def precision_recall(y_true, y_pred, positive):
    t = np.asarray(y_true) == positive
    p = np.asarray(y_pred) == positive
    tp = (t & p).sum()
    prec = tp / p.sum() if p.sum() else 0.0
    rec = tp / t.sum() if t.sum() else 0.0
    return float(prec), float(rec)
