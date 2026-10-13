"""Pick hyperparameters on TRAIN only (k-fold cross-validation)

Course: exam Ex1 (Feb 2026: "you may not use the test set as a validation
set"), Tema 2. Choosing C by looking at test accuracy is cheating and
overestimates your score. Also: fit the scaler inside each fold, or
information leaks from the validation fold into training.

TASK
  cv_scores(X, y, Cs, folds) -> dict {C: mean validation accuracy}
      folds: list of (train_idx, val_idx) pairs (given to you).
      For each C and each fold: fit a StandardScaler on the fold's TRAIN part
      only, transform both parts, train LinearSVC(C=C, max_iter=5000,
      random_state=0) and measure accuracy on the val part. Average over folds.
  best_C(scores) -> the C with the highest score (smallest C on ties).

ALLOWED  sklearn.preprocessing.StandardScaler, sklearn.svm.LinearSVC.
RULES    no GridSearchCV, cross_val_score, cross_validate, Pipeline.
"""


def cv_scores(X, y, Cs, folds):
    raise NotImplementedError


def best_C(scores):
    raise NotImplementedError
