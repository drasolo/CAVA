# How to approach any CAVA practical problem

Classify a face, read an X-and-0 board, count dots, score a Qwirkle grid — they are all
**the same pipeline**. You only change (a) *which stages do the heavy lifting* and
(b) *the numbers in the `PARAMS` block*. Learn the pipeline once, adapt per problem.

---

## The one pipeline (9 stages)

```
 0. CONFIG      every tunable number lives in PARAMS — and nowhere else
 1. LOAD        read image(s); read the annotation files (needed to calibrate)
 2. PREPROCESS  gray / HSV, blur, resize, normalize
 3. LOCALIZE    find the Region Of Interest (the board / the object)
 4. RECTIFY     warp the ROI to a straight, fixed-size view
 5. SEGMENT     cut the ROI into units (grid cells, or candidate windows)
 6. DESCRIBE    turn each unit into numbers (HOG / colour / blob count)
 7. DECIDE      classify / count / threshold each unit
 8. OUTPUT      assemble the units into the required text format, write the file
 9. CALIBRATE   tune PARAMS on the TRAIN set, measure accuracy, repeat
```

Data flows `1→8`. Stage `9` is the loop you run by hand: look at train results,
change a number in `PARAMS`, re-run. **You never hardcode a number inside a stage** —
if you want to change behaviour, you change `PARAMS`. That is the "single place".

---

## Two problem types = the same pipeline, different switches

| Stage | **Type A — Classify** (exam Ex1, faces) | **Type B — Extract** (exam Ex2, X/0, dots, Qwirkle) |
|------|------------------------------------------|------------------------------------------------------|
| 1 LOAD       | images + labels            | images + structured annotations |
| 2 PREPROCESS | gray + **resize**          | gray/HSV + blur |
| 3 LOCALIZE   | **crop by given bbox**, or skip | **find board/object** (the hard part) |
| 4 RECTIFY    | — skip —                   | perspective warp (if photo is angled) |
| 5 SEGMENT    | — skip — (1 image = 1 unit)| split into grid cells / windows |
| 6 DESCRIBE   | **HOG**                    | per-cell: % ink, blob count, or HOG |
| 7 DECIDE     | **SVM / kNN**              | rule (threshold / circularity) or SVM |
| 8 OUTPUT     | accuracy                   | the exact text format from the statement |
| 9 CALIBRATE  | pick `svm_C`               | tune thresholds on train images |

> Type A is just Type B **with stages 4 and 5 removed**. That is the whole "generalization".

---

## The raw concepts → where each one plugs in

| Concept (from the course)        | Stage      | Call you actually write                          | Param(s) in `PARAMS`            |
|----------------------------------|------------|--------------------------------------------------|---------------------------------|
| Colour convert (BGR→Gray/HSV)    | 2          | `cv2.cvtColor`                                   | —                               |
| Gaussian blur                    | 2          | `cv2.GaussianBlur`                               | `blur_ksize`                    |
| Resize                           | 2 / 4      | `cv2.resize`                                     | `work_size`                     |
| Colour mask                      | 3          | `cv2.inRange`                                    | `hsv_lo`, `hsv_hi`              |
| Canny edges                      | 3          | `cv2.Canny`                                      | `canny_lo`, `canny_hi`          |
| Contours                         | 3 / 5      | `cv2.findContours`                               | —                               |
| Polygon approximation            | 3          | `cv2.approxPolyDP`                               | `approx_eps_frac`               |
| Largest contour (pick the board) | 3          | `max(cnts, key=cv2.contourArea)`                 | `min_board_area`                |
| Hough lines / circles            | 3 / 7      | `cv2.HoughLinesP` / `cv2.HoughCircles`           | `hough_*`, `circle_*`           |
| Perspective transform            | 4          | `getPerspectiveTransform` + `warpPerspective`    | `work_size`                     |
| Morphology (clean up mask)       | 2 / 6      | `cv2.morphologyEx` (open/close)                  | `morph_ksize`                   |
| Threshold (Otsu / adaptive)      | 6 / 7      | `cv2.threshold` / `adaptiveThreshold`            | `occupied_white_pct`            |
| Circularity (tell `0` from `X`)  | 7          | `4*pi*area / perimeter**2`                       | `circularity_thresh`            |
| Blob / dot counting              | 6 / 7      | `findContours` + area/shape filter               | `dot_area_min/max`              |
| HOG features                     | 6          | `skimage.feature.hog`                            | `hog_orient`, `hog_ppc`, `hog_cpb` |
| Template / Chamfer matching      | 7          | `cv2.matchTemplate` / `distanceTransform`        | `match_thresh`                  |
| SVM / kNN classifier             | 7          | `LinearSVC` / `KNeighborsClassifier`             | `svm_C` / `knn_k`               |

If you can say, for a new problem, *"stage 3 = largest contour, stage 5 = 3×3 grid,
stage 7 = circularity rule"*, you have already solved it on paper. The skeleton just
fills those choices in.

---

## Using it in the exam (the 30-second plan)

1. Read the statement → fill the **OUTPUT format** (stage 8) first, so you know the target.
2. Decide the **type** (A classify / B extract) → cross out the stages you don't need.
3. For each remaining stage, pick **one concept** from the table above.
4. Open `skeletons/_template.py`, set the **paths + a first guess of PARAMS**.
5. Run stage 9 on **train** images, watch the printed accuracy, nudge `PARAMS` until good.
6. Switch to **test**, write output files, zip, submit.

See `skeletons/_template.py` — it is this document turned into code.
