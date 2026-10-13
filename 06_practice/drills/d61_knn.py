"""k-nearest neighbours from scratch

Course: Lab 11, exam Ex1 ("any classifier you want"). The simplest classifier,
and a good sanity check: if kNN on your features is bad, the features are bad.

TASK
  knn_predict(X_train, y_train, X_test, k) -> array of predicted labels
      Euclidean distance, majority vote among the k nearest training points.
      On a tie in the vote, pick the smallest label (what sklearn does).

RULES  no sklearn, scipy.
HINTS  hidden on purpose: python check.py d61 --hint (one step at a time)
"""


def knn_predict(X_train, y_train, X_test, k):
    raise NotImplementedError
