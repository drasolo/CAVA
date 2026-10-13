"""Blank exam notebooks for a mock attempt: headings, paths and the required
output format, but none of the solution."""
from pathlib import Path

import nbformat as nbf

from exams import EXAMS_SPEC

PRACTICE = Path(__file__).resolve().parent.parent

EX2_FORMAT = {
    "xo": (
        "For each test image `NNN.jpg` write `outputs/ex2/NNN.txt`, same format as the annotations:\n"
        "first line = number of filled cells, then one line per filled cell `position symbol`, "
        "positions 1..9 row by row, symbol `X` or `0` (the digit zero).\n\n"
        "Task 1 = all positions right; Task 2 = all symbols right. No partial credit per image."
    ),
    "single": (
        "For each test image `NNN.jpg` write `outputs/ex2/NNN.txt` containing just the result "
        "of the addition (e.g. `11`). One image right = 0.45 points."
    ),
    "ring": (
        "For each test image `NNN.jpg` write two files, like the annotations:\n"
        "- `outputs/ex2/NNN_1.txt`: 16 lines `k state`, state `o` (occupied) or `s` (free);\n"
        "- `outputs/ex2/NNN_2.txt`: 16 lines `k element`, element a digit 0-9, `+`, `-`, `=`, or `s`.\n\n"
        "Squares are numbered 1..16 CLOCKWISE starting at the top-left corner (it's the border of a 5x5 grid). "
        "Stage 1 = 0.15, stage 2 = 0.3 points per image, all squares right."
    ),
}


def _md(s):
    return nbf.v4.new_markdown_cell(s)


def _code(s):
    return nbf.v4.new_code_cell(s)


def ex1_notebook(exam):
    from cava_practice.faces import SPECS
    classes = sorted(set(SPECS[exam]["classes"].values()))
    return [
        _md(f"# Ex1 - {EXAMS_SPEC[exam]['title']}\n\n"
            "Re-read the train/test split rule in `Enunt.pdf` before writing code: a wrong split scores 0.\n"
            "Explain in comments which features and which classifier you use (required by the statement)."),
        _code("from pathlib import Path\nimport numpy as np\nimport cv2\n\nDATA = Path('data/ex1')\nOUT = Path('outputs')"),
        _code("# 1. Read the annotations (if any) and build TRAIN and TEST exactly as the statement says.\n"),
        _code("# 2. Features. Faces/images have different sizes: resize to a fixed size first.\n"),
        _code("# 3. Train the classifier. Tune only with cross-validation on TRAIN, never on TEST.\n"),
        _code("# 4. Accuracy on TEST.\n"),
        _md("## For the mock grader\n"
            f"Write `outputs/ex1_predictions.txt`: one line per TEST image, `<file name as in the data> <class>`, "
            f"with class one of: `{'`, `'.join(classes)}`."),
        _code("# 5. Write outputs/ex1_predictions.txt\n"),
    ]


def ex2_notebook(exam):
    kind = EXAMS_SPEC[exam]["ex2"]["kind"]
    return [
        _md(f"# Ex2 - {EXAMS_SPEC[exam]['title']}\n\n" + EX2_FORMAT[kind]),
        _code("from pathlib import Path\nimport numpy as np\nimport cv2\n\n"
              "TRAIN = Path('data/ex2/antrenare')\nTEST = Path('data/ex2/testare')\nOUT = Path('outputs/ex2')"),
        _code("# 1. Look at a few training images and their annotation files first.\n"),
        _code("# 2. Your pipeline: localise -> rectify -> cut into units -> decide -> format.\n"),
        _code("# 3. Run it on every TRAIN image and compare with the annotations.\n"
              "#    Write your own comparison first (in the exam you'll have to). Then, to check it, you can use:\n"
              f"# import sys; sys.path.insert(0, r'{PRACTICE}')\n"
              "# from cava_practice import exam_scoring   # score_xo / score_single / score_ring\n"),
        _code("# 4. Run on TEST and write the files to OUT.\n"),
    ]


def write_notebooks(att, exam):
    import sys
    sys.path.insert(0, str(PRACTICE))
    for name, cells in (("ex1.ipynb", ex1_notebook(exam)), ("ex2.ipynb", ex2_notebook(exam))):
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nbf.write(nb, str(att / name))
