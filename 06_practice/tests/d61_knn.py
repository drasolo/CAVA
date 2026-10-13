import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from cava_practice.testing import check, forbid


def test(m):
    forbid(m, ["sklearn", "scipy"])
    X, y = make_classification(n_samples=300, n_features=20, n_informative=6, n_classes=3,
                               n_clusters_per_class=1, random_state=0)
    Xtr, ytr, Xte = X[:200], y[:200], X[200:]
    for k in (1, 3, 5, 9):
        ref = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr).predict(Xte)
        got = np.asarray(m.knn_predict(Xtr, ytr, Xte, k))
        check(got.shape == ref.shape, f"k={k}: predictions shape {got.shape}, expected {ref.shape}")
        agree = (got == ref).mean()
        check(agree >= 0.99, f"k={k}: agrees with sklearn on only {agree:.1%} of test points")
    labels = np.array(["copii", "parinti"])[y[:200] % 2]
    got = m.knn_predict(Xtr, labels, Xte[:5], 3)
    check(all(g in ("copii", "parinti") for g in got), "knn_predict must work with string labels too")
