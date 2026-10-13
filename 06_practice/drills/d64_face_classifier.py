"""Your first real classifier: four past exams, one pipeline

Course: exam Ex1 (every session). This drill gives you the faces already
cropped and split exactly as each statement requires (cava_practice/faces.py).
Your job is the part the exam grades: features + classifier. The same two
functions must reach >= 85% test accuracy on ALL FOUR past exams:
  2025-02-06  Dexter's Lab kids vs Louie's show kids (whole images)
  2025-02-20  parents vs kids (Dexter's Lab)
  2025-06-06  Andy+Louie vs Ora+Tommy (family resemblance: the hard one)
  2026-02-06  Daphne+Velma vs Fred+Shaggy (Scooby-Doo)

TASK
  features(face_bgr) -> 1D numpy vector, same length for every face.
      Faces have different sizes: resize to a fixed size first.
  make_classifier() -> an unfitted sklearn estimator (fit/predict).

RULES  Hyperparameters are fixed in code. Don't tune them by looking at test
       accuracy: that's what d62 is for (cross-validate on the training set).
IDEAS  HOG (d50) on a 64x64 grayscale face; HSV histogram (d51); both
       concatenated; StandardScaler + LinearSVC or SVC. Print the confusion
       matrix (d63) on the exam that fails to see what's going wrong.
"""


def features(face_bgr):
    raise NotImplementedError


def make_classifier():
    raise NotImplementedError
