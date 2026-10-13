import numpy as np


def knn_predict(X_train, y_train, X_test, k):
    Xtr = np.asarray(X_train, np.float64)
    Xte = np.asarray(X_test, np.float64)
    ytr = np.asarray(y_train)
    d = (Xte ** 2).sum(1)[:, None] + (Xtr ** 2).sum(1)[None, :] - 2 * Xte @ Xtr.T
    nn = np.argsort(d, axis=1, kind="stable")[:, :k]
    classes = np.unique(ytr)
    out = []
    for row in nn:
        votes = [(ytr[row] == c).sum() for c in classes]
        out.append(classes[int(np.argmax(votes))])
    return np.array(out)
