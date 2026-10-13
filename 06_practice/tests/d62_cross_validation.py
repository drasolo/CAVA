import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from cava_practice.testing import check, forbid


def test(m):
    forbid(m, ["GridSearchCV", "cross_val_score", "cross_validate", "Pipeline", "make_pipeline"])
    X, y = make_classification(n_samples=160, n_features=40, n_informative=8, random_state=3)
    X[:, :5] *= 100  # wildly different feature scales: scaling matters
    Cs = [0.001, 0.01, 0.1, 1.0]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    pipe = make_pipeline(StandardScaler(), LinearSVC(max_iter=5000, random_state=0))
    gs = GridSearchCV(pipe, {"linearsvc__C": Cs}, cv=folds).fit(X, y)
    ref = dict(zip(Cs, gs.cv_results_["mean_test_score"]))
    got = m.cv_scores(X, y, Cs, folds)
    check(set(got) == set(Cs), f"cv_scores must have one entry per C, got keys {sorted(got)}")
    for C in Cs:
        check(abs(got[C] - ref[C]) < 1e-9,
              f"C={C}: your CV accuracy {got[C]:.4f}, expected {ref[C]:.4f}. "
              "Is the scaler fitted on the fold's training part only?")
    check(m.best_C({0.1: 0.8, 1.0: 0.9, 10.0: 0.9}) == 1.0, "best_C: highest score, smallest C on ties")
