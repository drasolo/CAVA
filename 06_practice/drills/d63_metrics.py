"""Accuracy, confusion matrix, precision, recall

Course: exam Ex1 is graded on accuracy; Tema 2 on precision/recall (AP). The
confusion matrix tells you WHICH classes get mixed up, which is what you need
to improve anything.

TASK
  accuracy(y_true, y_pred) -> float
  confusion_matrix(y_true, y_pred, labels) -> int array (L x L)
      row = true label, column = predicted label, in the order of `labels`.
  precision_recall(y_true, y_pred, positive) -> (precision, recall)
      for one class treated as positive. Return 0.0 instead of dividing by 0.

RULES  no sklearn.
"""


def accuracy(y_true, y_pred):
    raise NotImplementedError


def confusion_matrix(y_true, y_pred, labels):
    raise NotImplementedError


def precision_recall(y_true, y_pred, positive):
    raise NotImplementedError
