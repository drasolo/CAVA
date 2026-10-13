import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def cv_scores(X, y, Cs, folds):
    X, y = np.asarray(X), np.asarray(y)
    out = {}
    for C in Cs:
        accs = []
        for tr, va in folds:
            sc = StandardScaler().fit(X[tr])
            clf = LinearSVC(C=C, max_iter=5000, random_state=0).fit(sc.transform(X[tr]), y[tr])
            accs.append((clf.predict(sc.transform(X[va])) == y[va]).mean())
        out[C] = float(np.mean(accs))
    return out


def best_C(scores):
    return min(scores, key=lambda c: (-scores[c], c))
