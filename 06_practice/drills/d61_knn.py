"""k-nearest neighbours from scratch

Course: Lab 11, exam Ex1 ("any classifier you want"). The simplest classifier,
and a good sanity check: if kNN on your features is bad, the features are bad.

TASK
  knn_predict(X_train, y_train, X_test, k) -> array of predicted labels
      Euclidean distance, majority vote among the k nearest training points.
      On a tie in the vote, pick the smallest label (what sklearn does).

RULES  no sklearn, scipy.
HINT   All pairwise squared distances at once:
       |a - b|^2 = |a|^2 + |b|^2 - 2 a.b   ->  (n_test, n_train) matrix.
       np.argpartition(d, k, axis=1)[:, :k] finds the k smallest without a full sort.
"""


def knn_predict(X_train, y_train, X_test, k):
    raise NotImplementedError
