import cv2
import numpy as np
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SIZE = (64, 64)


def features(face_bgr):
    face = cv2.resize(face_bgr, SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    f_hog = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    f_col = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return np.r_[f_hog, 3 * f_col / f_col.sum()]


def make_classifier():
    return make_pipeline(StandardScaler(), LinearSVC(C=0.01, dual=True, max_iter=5000))
