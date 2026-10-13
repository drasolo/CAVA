# CAVA practice

Three ways to train, from small to exam-sized:

| Tool | What it trains | Run |
|---|---|---|
| **Drills** (`drills/`) | One concept at a time. You fill in a function; `check.py` tests it against OpenCV / scikit-learn / real annotations and tells you exactly what's wrong. | `python check.py next` |
| **Mock exams** (`mocks/`) | The real 2-hour lab exam on past subjects, with a sealed test set and the statement's scoring. | `python mocks/start.py` |
| **Theory quiz** (`theory.py`) | The written part. Questions you miss come back more often. | `python theory.py` |

Run everything from this folder (`06_practice`). The exam data comes from the backup mirror on D:
(see `cava_practice/paths.py`; set `CAVA_BACKUP` if it moves). Drills d00 to d63 and d70 to d73 need
no data; d64 and the mocks do.

## How a drill works

```
python check.py              # list all drills and your progress
python check.py d12          # test drills/d12_sobel.py
python check.py next         # test the first drill you haven't passed
python check.py d12 --ref    # run the reference solution (proof the bar is reachable)
```

1. Open `drills/dNN_*.py`. The docstring has the course link, the task, the rules, and hints.
2. Write the functions. "From scratch" drills forbid the library call they reimplement; the
   checker reads your code and refuses it if you use one.
3. `python check.py dNN` until it says PASS. Failures name the case, the value you got, and the
   value expected.
4. Only then read `solutions/dNN_*.py` and compare. If you're stuck for more than ~45 minutes,
   read the solution, close it, and rewrite yours from memory the next day.

Progress is saved in `progress.json` (date passed, number of attempts).

## The curriculum

Each level builds on the one before. The right-hand column is where it shows up in the course.

| Level | Drills | Concepts | Used in |
|---|---|---|---|
| 0 Images | d00 gray · d01 crop · d02 uint8 · d03 resize | BGR order, inclusive boxes, uint8 overflow, (w, h) vs (h, w) | everything |
| 1 Filters | d10 correlate · d11 gaussian · d12 sobel · d13 median | correlation vs convolution, separability, gradients, denoising | Curs 2-4 |
| 2 Edges | d20 edge NMS · d21 Hough lines · d22 RANSAC | Canny's thinning, voting in (rho, theta), robust fitting, the N formula | Curs 4-5, board lines |
| 3 Binary | d30 Otsu · d31 morphology · d32 components · d33 X/0/empty · d34 HSV colours | thresholds, opening/closing, 4 vs 8 connectivity, shape rules, hue wrap | Curs 6-7, exam Ex2, Qwirkle |
| 4 Geometry | d40 corner order · d41 homography · d42 grid numbering · d43 find board | perspective, cell indexing (row-major, clockwise ring, Qwirkle "2H") | exam Ex2, Sudoku, Qwirkle |
| 5 Features | d50 HOG · d51 colour histogram · d52 NCC · d53 count dots | the descriptors you'll classify with; template matching; blob counting | Curs 8, exam Ex1/Ex2, Tema 2 |
| 6 Learning | d60 exam split · d61 kNN · d62 cross-validation · d63 metrics · d64 face classifier | reproducing splits exactly, tuning on train only, confusion matrices | exam Ex1, Tema 2 |
| 7 Detection | d70 IoU · d71 NMS · d72 sliding windows · d73 average precision | the Tema 2 detector and its grader | Lab 9-10, Tema 2 |

d64 is the bridge to the exam: one feature extractor + classifier that must reach 85% on the
official split of all four past Ex1 subjects.

## Mock exams

```
python mocks/start.py                    # list the exams
python mocks/start.py 2025-06-06         # new attempt folder, clock starts
python mocks/grade.py mocks/attempts/2025-06-06_<stamp>
```

| Exam | Ex1 | Ex2 |
|---|---|---|
| 2025-02-06 | Dexter's Lab kids vs Louie's kids, whole images | X&0 on a wooden board |
| 2025-02-20 | parents vs kids, crop by annotation | X&0 on a red plastic toy |
| 2025-06-06 | Andy+Louie vs Ora+Tommy, names in the annotation file | dice: sum of 3 squares |
| 2026-02-06 | Daphne+Velma vs Fred+Shaggy | 16-square math ring, 2 stages |

The attempt folder looks like the exam room: `Enunt.pdf`, `data/`, `ex1.ipynb`, `ex2.ipynb`. The
real Ex2 test sets were never published, so the last ~40% of each Ex2 training set is sealed as
your test set (images in `data/ex2/testare`, answers in `.sealed/`, which you don't open). That
leaves less training data than the real exam had, so the mock is slightly harder.

Rules that make the score honest:
- Time-box it: 2 hours, statement read included. The grader prints the time since start.
- No opening your old attempts or the `05_exam-prep` skeletons during the attempt. Afterwards,
  compare with them.
- After grading, fix what failed without the clock, then **redo the same exam from scratch a week
  later**. The second attempt is the one that shows what stuck.

Traps already found in the real data (the grader and loaders handle them; your exam code must too):
- 2025-06-06 Ex1: the annotation file says `001.jpg` but the files are `001.png`. `cv2.imread`
  returns `None` silently.
- 2026-02-06 Ex2: the 16 squares are the border of a 5x5 grid, numbered clockwise from the
  top-left corner. Not row-major.
- X&0 annotations use the digit `0`, not the letter `O`.

## Theory quiz

```
python theory.py         # 10 questions
python theory.py 25
python theory.py --stats # per-topic hit rate, weakest first
```

Answer out loud or on paper before revealing. ~50 questions from Curs 1-12: filters, Canny,
Hough, RANSAC (with the formula), morphology, HSV, shape matching, homographies, HOG, detection
and AP, bag of words, SIFT, CNN arithmetic, autoencoders, seam carving, background subtraction.

## A weekly routine

- **3 to 4 short sessions:** 1-2 drills each, in order. Then 10 theory questions.
- **1 long session:** a mock exam (once you're past level 5), or redo a failed one.
- Every drill you pass, explain in two sentences what it does and why. If you can't, it hasn't
  stuck yet.

## What's not built yet

Next batches, in the order I'd add them:
1. **Lab projects:** seam carving (DP), mosaic, background subtraction on video.
2. **Tema-scale:** rebuild the Qwirkle pipeline (frame differencing, tile classification, the
   scoring engine as a pure-logic drill against the annotations) and the Scooby detector (hard
   negatives, multi-scale, AP on validation).
3. **ML track:** PyTorch install, CNN features for Ex1, a small CNN on cells, YOLO for the
   Tema 2 bonus.
4. **Feb 2026b mock** (faces vs 40x40 background patches + car pictograms): its data isn't in
   the backup, only the statement.
