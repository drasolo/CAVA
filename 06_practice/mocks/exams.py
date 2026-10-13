"""The past lab exams available as mocks.

Ex2 test sets were never published to you, so each mock seals the last ~40%
of the Ex2 training images as its test set: you get the images without
annotations, the grader keeps the answers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cava_practice.paths import EXAMS, REPO  

EXAMS_SPEC = {
    "2025-02-06": {
        "title": "Lab exam 6 Feb 2025: Dexter's Lab vs Louie (whole images) + X&0 on a wooden board",
        "statement": REPO / "02_exam-jun2025" / "subjects" / "Enunt.pdf",
        "ex2": {"kind": "xo", "task_points": (2.5, 2.5)},
    },
    "2025-02-20": {
        "title": "Lab exam 20 Feb 2025: parents vs kids (crop by annotation) + X&0 on a red toy board",
        "statement": REPO / "05_exam-prep" / "practice-subjects" / "Subiect2" / "Enunt.pdf",
        "ex2": {"kind": "xo", "task_points": (2.0, 1.5)},
    },
    "2025-06-06": {
        "title": "Lab exam 6 Jun 2025: Andy+Louie vs Ora+Tommy (names in annotations) + dice sums",
        "statement": REPO / "02_exam-jun2025" / "subjects" / "Subiect3" / "Subiect3" / "Enunt.pdf",
        "ex2": {"kind": "single", "task_points": (4.5,)},
    },
    "2026-02-06": {
        "title": "Lab exam 6 Feb 2026: Daphne+Velma vs Fred+Shaggy + 16-square math ring (2 stages)",
        "statement": REPO / "05_exam-prep" / "attempt_Lucrare1" / "Lucrare1" / "Enunt.pdf",
        "ex2": {"kind": "ring", "task_points": (1.5, 3.0)},
    },
}

SEALED_FRACTION = 0.4


def ex2_train_dir(exam):
    return EXAMS[exam] / "ex2" / "antrenare"


def ex2_images(exam):
    return sorted(p for p in ex2_train_dir(exam).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def ex2_annotations(exam, img):
    """Annotation files belonging to one image (one for X&0 / dice, two for the ring)."""
    d = img.parent
    if EXAMS_SPEC[exam]["ex2"]["kind"] == "ring":
        return [d / f"{img.stem}_1.txt", d / f"{img.stem}_2.txt"]
    return [d / f"{img.stem}.txt"]


def ex2_split(exam):
    imgs = ex2_images(exam)
    n_sealed = max(1, round(len(imgs) * SEALED_FRACTION))
    return imgs[:-n_sealed], imgs[-n_sealed:]
