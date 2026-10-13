"""Exam Ex1 face datasets with their OFFICIAL train/test splits.

split(exam)  -> list of Item(name, box, label, part) without reading images;
                part is "train", "test" or None (unused).
load(exam)   -> (train, test): lists of (face_bgr, label), cropped with the
                annotations where the statement requires it.
"""
import os
from collections import Counter, namedtuple

import cv2

from .paths import EXAMS, require

Item = namedtuple("Item", "name box label part")

# Per exam: where ex1 lives, the binary class of each character, points.
SPECS = {
    "2025-02-06": {"classes": {"deedee": "dexter_lab", "dexter": "dexter_lab", "louie": "louie_show", "tommy": "louie_show"},
                   "points": 4.0},
    "2025-02-20": {"classes": {"mom": "parents", "dad": "parents", "dexter": "kids", "deedee": "kids"},
                   "points": 5.5},
    "2025-06-06": {"classes": {"andy": "andy_louie", "louie": "andy_louie", "ora": "ora_tommy", "tommy": "ora_tommy"},
                   "points": 4.5},
    "2026-02-06": {"classes": {"daphne": "female", "velma": "female", "fred": "male", "shaggy": "male"},
                   "points": 4.5},
}


def ex1_dir(exam):
    return require(EXAMS[exam] / "ex1")


def _read_annotations(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if p:
                rows.append((p[0], tuple(int(float(v)) for v in p[1:5]), p[5].lower() if len(p) > 5 else None))
    return rows


def _annotation_file(d):
    return next(p for p in d.iterdir() if p.suffix == ".txt")


def _by_ranges(name, ranges):
    num = int(os.path.splitext(name)[0])
    for lo, hi, who in ranges:
        if lo <= num <= hi:
            return who
    raise ValueError(name)


def split(exam):
    d = ex1_dir(exam)
    cls = SPECS[exam]["classes"]
    items = []
    if exam == "2025-02-06":  # whole images, blocks of 55, first 40 train
        chars = [(1, 55, "deedee"), (56, 110, "louie"), (111, 165, "dexter"), (166, 220, "tommy")]
        for i in range(1, 221):
            name = f"{i:04d}.jpg"
            items.append(Item(name, None, cls[_by_ranges(name, chars)], "train" if (i - 1) % 55 < 40 else "test"))
    elif exam == "2025-02-20":  # crop; blocks of 50, first 40 train
        chars = [(1, 50, "mom"), (51, 100, "dexter"), (101, 150, "dad"), (151, 200, "deedee")]
        for name, box, _ in _read_annotations(_annotation_file(d)):
            i = int(os.path.splitext(name)[0])
            items.append(Item(name, box, cls[_by_ranges(name, chars)], "train" if (i - 1) % 50 < 40 else "test"))
    elif exam == "2025-06-06":  # crop; names in file; first 45 per character train, next 10 test
        seen = Counter()
        for name, box, who in _read_annotations(_annotation_file(d)):
            k = seen[who]
            seen[who] += 1
            items.append(Item(name, box, cls[who], "train" if k < 45 else "test" if k < 55 else None))
    elif exam == "2026-02-06":  # crop; character by range; first 40 per character in file order
        chars = [(1, 55, "daphne"), (56, 110, "fred"), (111, 165, "velma"), (166, 220, "shaggy")]
        seen = Counter()
        for name, box, _ in _read_annotations(_annotation_file(d)):
            who = _by_ranges(name, chars)
            k = seen[who]
            seen[who] += 1
            items.append(Item(name, box, cls[who], "train" if k < 40 else "test"))
    else:
        raise KeyError(exam)
    return items


def imread_any(folder, name):
    """Read `name`, falling back to the same stem with another extension: the
    Jun 2025 annotations say 001.jpg but the files are 001.png, and
    cv2.imread returns None instead of raising."""
    path = folder / name
    if not path.exists():
        path = next(folder.glob(os.path.splitext(name)[0] + ".*"))
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"could not read {path}")
    return img


def _crop(img, box):
    if box is None:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[max(0, y0):min(h, y1 + 1), max(0, x0):min(w, x1 + 1)]


def load(exam):
    d = ex1_dir(exam)
    train, test = [], []
    for it in split(exam):
        if it.part is None:
            continue
        face = _crop(imread_any(d, it.name), it.box)
        (train if it.part == "train" else test).append((face, it.label))
    return train, test


ALL = {exam: (lambda e=exam: load(e)) for exam in SPECS}
