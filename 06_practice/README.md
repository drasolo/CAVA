# CAVA practice

## Start here

```
cava                 # from any PowerShell window (or the "cava coach" desktop shortcut)
python coach.py      # the same thing, from this folder
```

The first time, the coach offers the **onboarding** (`python learn.py`): one drill, d00, start to
finish with the whole learning flow. After that, open the coach, say how you feel, and it picks
the next step and starts it. `python coach.py --list` shows every option:

| You feel | The coach does |
|---|---|
| I can't get started | a 5-minute starter: only the "predict" step of your current drill |
| Stuck for 15+ minutes | the next hint step (strict drills unlock hints after 15 minutes) |
| My output differs and I don't see why | re-runs the checker on the failing case (visual diff: phase 2) |
| I don't get the concept | walks the concept map down to something you have, then back up |
| Overwhelmed | shrinks it: one function, its part of the task, a 10-minute goal |
| Bored / tired / only 15 minutes | review items: due cards first, then theory |
| Anxious about the exam | a timed from-memory redo of a drill you passed |
| On fire | the next drill, or the redo that is due |
| Just finished a drill | teach-back: Claude asks you one probing question |

A **session** starts with your first action. A small always-on-top window shows the active time
(it pauses after 10 idle minutes; saving a drill file or running the checker counts as activity).
It turns green at 20 minutes: shorter sessions don't count. Under it is the week's total against
the target (4 hours, in `me/config.json`). When a counted session ends, its minutes go to the
`cava` habit on dra-solo and `me/` is pushed to its private repo.

## The flow

Every drill runs inside the same loop. Each step is there because it makes learning stick:

1. **Read the task, not the hints.** The hints live in `hints/`, not in the drill file.
2. **Predict** before coding: your plan, the trap you expect, one concrete value.
3. **Write it cold.** Docs yes; solutions, old code and AI no.
4. **Before each run, say what you expect. After a failure, write why** before the next run.
5. **Hints on a timer:** `python check.py dNN --hint`, one step at a time.
6. **Compare** with `solutions/` only after you pass.
7. **Explain it in two sentences**: what it does, why it matters later.
8. **Turn the trap into a card.** It joins the review queue (`python theory.py`).
9. **Redo it from memory** later: `python check.py dNN --redo` archives your code to `me/archive/`,
   brings the blank stub back, and times you. It sticks when it passes first time in a third of
   the original time; otherwise it comes back the next day.

The **first 10 drills are strict**: the prediction, the per-run expectation, the failure note and
the card are required, hints unlock after 15 minutes, and redos come after 1 day, 1 week and a week
before the exam. After that the flow is light (only the two sentences are required; one redo after
3 days, one before the exam). The strict phase ends early after three strict drills in a row passed
in two runs or fewer with a correct prediction.

```
python check.py              # list all drills, your progress, due redos ([~])
python check.py d12          # run drills/d12_sobel.py inside the flow
python check.py d12 --hint   # the next hint step
python check.py d12 --redo   # start a from-memory redo
python check.py next         # a due redo, else the first drill not passed
python check.py d12 --ref    # run the reference solution (proof the bar is reachable)
python selftest.py           # tests for this tooling
```

"From scratch" drills forbid the library call they reimplement; the checker reads your code and
refuses it if you use one. `CAVA_FLOW=off` runs a drill with no questions asked.

## Your files: `me/`

Everything personal lives in `me/`, which is gitignored here and is its own private repo
(`drasolo/cava-progress`): `progress.json` (every run, prediction, note and redo), `cards.json`,
`theory_progress.json` (the review schedule), `sessions.json`, `config.json` (weekly target, Tema
milestones, exam date), `flags.json` (concept explanations that didn't help) and `journal.md`, a
revision sheet rebuilt after every run. Move it with `CAVA_ME`.

Three ways to train, from small to exam-sized:

| Tool | What it trains | Run |
|---|---|---|
| **Drills** (`drills/`) | One concept at a time. You fill in a function; `check.py` tests it against OpenCV / scikit-learn / real annotations and tells you exactly what's wrong. | `python check.py next` |
| **Mock exams** (`mocks/`) | The real 2-hour lab exam on past subjects, with a sealed test set and the statement's scoring. | `python mocks/start.py` |
| **Review** (`theory.py`) | The written part plus your cards, on a Leitner schedule (1 day, 3 days, 1 week, 2 weeks, 1 month). | `python theory.py` |

Run everything from this folder (`06_practice`). The exam data comes from the backup mirror on D:
(see `cava_practice/paths.py`; set `CAVA_BACKUP` if it moves). Drills d00 to d63 and d70 to d73 need
no data; d64 and the mocks do.

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

## Review

```
python theory.py         # up to 10 items: due first, then new
python theory.py 25
python theory.py --due   # how many are due today
python theory.py --stats # per-topic hit rate, weakest first
```

Answer out loud or on paper before revealing. ~50 questions from Curs 1-12 (filters, Canny,
Hough, RANSAC with the formula, morphology, HSV, shape matching, homographies, HOG, detection
and AP, bag of words, SIFT, CNN arithmetic, autoencoders, seam carving, background subtraction),
plus one card per drill you pass.

## A weekly routine

- **4 hours a week** to start (the coach suggests adjusting after two weeks). Tema work counts.
- **3 to 4 sessions of 30 to 50 minutes:** 1-2 drills each, in order, plus one older redo.
- **1 long session:** a mock exam (once you're past level 5), a capstone, or Tema work.
- **Pace:** levels 0 to 4 before Tema 1 is published (~10 November), levels 5 to 7 before
  Tema 2 (mid-December). The coach shows where you stand. Change the dates in `me/config.json`
  when the real statements come out.

## What's next

1. **Help when stuck:** `check.py --show` (your image, the expected one, a diff heatmap), the live
   tutor in the coach, concept maps for drills 11 to 33, a webcam playground for passed drills.
2. **Fun and rehearsal:** a `mycv` library of your passed drills, capstones that rebuild last
   year's Qwirkle grid and board with it, bug hunts, speedrun personal bests, a progress map
   with bosses.
3. **Exam (January):** Lucrare3 (5 June 2026) as a mock, drills for pictograms-by-legend and
   face vs non-face, a submission check (zip name, fresh-kernel run).
4. **Later:** lab projects (seam carving, mosaic, background subtraction), an ML track (CNN
   features, YOLO for the Tema 2 bonus), the Feb 2026b mock if its data turns up.
