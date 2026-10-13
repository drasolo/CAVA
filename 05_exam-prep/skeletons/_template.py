"""
CAVA — generalized solution skeleton.   (read ../APPROACH.md first)

ONE pipeline:  LOAD -> PREPROCESS -> LOCALIZE -> RECTIFY -> SEGMENT -> DESCRIBE -> DECIDE -> OUTPUT
- Type A (classify): keep stages 2,6,7  (crop by bbox in 3, skip 4 & 5)
- Type B (extract) : use all stages

RULE: the only numbers you ever touch live in PARAMS below. Stage functions read PARAMS,
they never contain a magic number. Tune on TRAIN (stage 9), then run on TEST.

This is a skeleton: each stage shows the key concept + a `TODO`. Fill the TODOs.
Paste into a Jupyter cell at the exam if you prefer notebooks.
"""
import os
import cv2
import numpy as np
from dataclasses import dataclass

# skimage / sklearn only needed for the classification path:
# from skimage.feature import hog
# from sklearn.svm import LinearSVC


# ==========================================================================
# STAGE 0 — PARAMS  ·  THE SINGLE PLACE YOU CHANGE ANYTHING
# ==========================================================================
@dataclass
class Params:
    # --- paths (1 LOAD) -------------------------------------------------
    train_dir: str = "data/antrenare"
    test_dir:  str = "data/testare"
    out_dir:   str = "data/output"
    img_ext:   str = ".jpg"

    # --- 2 PREPROCESS ---------------------------------------------------
    blur_ksize: int = 5
    work_size:  tuple = (450, 450)        # canonical size after RECTIFY

    # --- 3 LOCALIZE -----------------------------------------------------
    canny_lo: int = 50
    canny_hi: int = 150
    approx_eps_frac: float = 0.02         # polygon tolerance (× perimeter)
    min_board_area:  int = 10000
    use_color_mask: bool = False          # True -> localize by colour instead of edges
    hsv_lo: tuple = (35, 40, 40)          # e.g. green board
    hsv_hi: tuple = (85, 255, 255)

    # --- 5 SEGMENT ------------------------------------------------------
    grid_rows: int = 3
    grid_cols: int = 3
    cell_margin: int = 8                  # px trimmed off each cell (kills grid lines)

    # --- 6 DESCRIBE / 7 DECIDE ------------------------------------------
    occupied_white_pct: float = 0.05      # cell counts as "filled" above this fraction
    circularity_thresh: float = 0.60      # >thr => round ('0' / dot), else 'X'
    dot_area_min: int = 20                # blob-count limits (dots / pips)
    dot_area_max: int = 3000

    # --- classification path (Type A) -----------------------------------
    clf_size:   tuple = (128, 128)
    hog_orient: int = 9
    hog_ppc:    tuple = (8, 8)
    hog_cpb:    tuple = (3, 3)
    svm_C:      float = 1.0


P = Params()   # <-- edit fields above, not the code below


# ==========================================================================
# STAGE 1 — LOAD
# ==========================================================================
def load_image(path):
    return cv2.imread(path)                       # BGR or None

def list_images(folder, p=P):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(p.img_ext))


# ==========================================================================
# STAGE 2 — PREPROCESS   (concepts: cvtColor, GaussianBlur)
# ==========================================================================
def preprocess(img, p=P):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (p.blur_ksize, p.blur_ksize), 0)
    return gray


# ==========================================================================
# STAGE 3 — LOCALIZE the ROI   (concepts: Canny/inRange, findContours, approxPolyDP)
# returns the 4 corner points of the board, or None
# ==========================================================================
def find_board_quad(img, gray, p=P):
    if p.use_color_mask:
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(p.hsv_lo), np.array(p.hsv_hi))
        edges = cv2.Canny(mask, p.canny_lo, p.canny_hi)
    else:
        edges = cv2.Canny(gray, p.canny_lo, p.canny_hi)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < p.min_board_area:
            break
        approx = cv2.approxPolyDP(c, p.approx_eps_frac * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None                                   # TODO: fallback if board not found


# ==========================================================================
# STAGE 4 — RECTIFY   (concepts: getPerspectiveTransform, warpPerspective)
# ==========================================================================
def _order_corners(pts):
    s, d = pts.sum(1), np.diff(pts, axis=1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)  # TL,TR,BR,BL

def rectify(img, quad, p=P):
    w, h = p.work_size
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(_order_corners(quad), dst)
    return cv2.warpPerspective(img, M, (w, h))


# ==========================================================================
# STAGE 5 — SEGMENT into units   (concept: grid slicing; or sliding window)
# returns dict {position -> cell image}, positions numbered 1..rows*cols
# ==========================================================================
def split_grid(board, p=P):
    h, w = board.shape[:2]
    ch, cw, m = h // p.grid_rows, w // p.grid_cols, p.cell_margin
    cells, pos = {}, 1
    for r in range(p.grid_rows):
        for c in range(p.grid_cols):
            y, x = r * ch, c * cw
            cells[pos] = board[y + m:y + ch - m, x + m:x + cw - m]
            pos += 1
    return cells


# ==========================================================================
# STAGE 6+7 — DESCRIBE + DECIDE per unit   (concepts: Otsu, circularity, blob count)
# ==========================================================================
def is_occupied(cell, p=P):
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (np.count_nonzero(th) / th.size) > p.occupied_white_pct

def classify_symbol(cell, p=P):
    """Tell a round shape ('0') from an 'X' via circularity of the biggest contour."""
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 'X'
    c = max(cnts, key=cv2.contourArea)
    per = cv2.arcLength(c, True)
    circ = 4 * np.pi * cv2.contourArea(c) / (per * per) if per else 0
    return '0' if circ > p.circularity_thresh else 'X'

def count_dots(cell, p=P):
    """Count pips/dots = circular blobs within an area range."""
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = 0
    for c in cnts:
        a = cv2.contourArea(c)
        if p.dot_area_min <= a <= p.dot_area_max:
            n += 1                                # TODO: add circularity filter if noisy
    return n


# ==========================================================================
# STAGE 8 — OUTPUT   (build the EXACT text format from the statement, then write)
# ==========================================================================
def solve_one(img, p=P):
    """Type B extraction. Returns the dict you will format. TODO: match the statement."""
    gray = preprocess(img, p)
    quad = find_board_quad(img, gray, p)
    board = rectify(img, quad, p) if quad is not None else cv2.resize(img, p.work_size)
    cells = split_grid(board, p)
    filled = {pos: classify_symbol(cell, p)       # or count_dots(cell, p)
              for pos, cell in cells.items() if is_occupied(cell, p)}
    return filled

def write_output(path, filled):
    with open(path, 'w') as f:                    # TODO: match the required format exactly
        f.write(f"{len(filled)}\n")
        for pos in sorted(filled):
            f.write(f"{pos} {filled[pos]}\n")


# ==========================================================================
# STAGE 9 — CALIBRATE + EVALUATE on TRAIN, then RUN on TEST
# ==========================================================================
def run(folder, p=P, write=False):
    os.makedirs(p.out_dir, exist_ok=True)
    for name in list_images(folder, p):
        img = load_image(os.path.join(folder, name))
        filled = solve_one(img, p)
        if write:
            write_output(os.path.join(p.out_dir, name.replace(p.img_ext, ".txt")), filled)
        else:
            print(name, filled)                   # eyeball vs the train annotation, tune PARAMS

if __name__ == "__main__":
    run(P.train_dir, P, write=False)              # 1) calibrate: look, tweak PARAMS, repeat
    # run(P.test_dir, P, write=True)              # 2) when happy: generate submission files


# ==========================================================================
# TYPE A (classification) — the reduced pipeline, for reference.
# Keep PREPROCESS + DESCRIBE(HOG) + DECIDE(SVM); skip LOCALIZE/RECTIFY/SEGMENT.
# ==========================================================================
# def to_features(gray, p=P):
#     from skimage.feature import hog
#     g = cv2.resize(gray, p.clf_size)
#     return hog(g, orientations=p.hog_orient, pixels_per_cell=p.hog_ppc,
#                cells_per_block=p.hog_cpb, block_norm='L2-Hys', transform_sqrt=True)
#
# def train_classifier(X, y, p=P):
#     from sklearn.svm import LinearSVC
#     clf = LinearSVC(C=p.svm_C, max_iter=5000); clf.fit(X, y); return clf
#
# # crop a face with the given bbox (stage 3 for Type A):  face = img[ymin:ymax, xmin:xmax]
