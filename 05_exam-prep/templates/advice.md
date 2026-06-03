## Minimal “what to change” when the task is **very similar** (grid board + per-cell labels)

Assume the pipeline stays:

**image → warp to square → split into N×N cells → preprocess cell → classify symbol/empty → write output**

If the new exercise is Sudoku / TicTacToe / similar, you typically only touch **5 things**.

---

# 1) Paths + output names (always)

**What to look for in the statement**

- where images are
- where training labels are
- required output folder + file naming

**What to change**

- `.env`: `EX2_DIR`, `EX2_TRAIN_DIR`, `EX2_TEST_DIR`, `EX2_OUT_DIR`
- suffix naming if they change: `EX2_ANN1_SUFFIX`, `EX2_ANN2_SUFFIX`

✅ Everything else can stay.

---

# 2) Grid size N (almost always)

**What to look for**

- “board is 9×9 / 3×3 / 4×4 …”

**What to change**

- `.env`: `EX2_GRID_N`
- and only if your code hard-assumes 4×4: update `get_number_map(n, scheme)` to support n≠4.

**Minimal rule**

- If they say “row-major numbering”:
  `id = r*N + c + 1`
- If they specify a special order (spiral, snake, etc.): add that map as a candidate.

---

# 3) Cell indexing / numbering scheme (the #1 reason exact-match fails)

**What to look for**

- “cells are numbered 1..N² in row-major order” OR “spiral” OR they show a diagram

**What to change**

- `get_number_map(...)` candidates (row_major vs spiral vs snake)
- keep the **auto-choose** evaluation on training (best practice)

**Heuristic**

- If exact-match is near 0, and your symbols look correct visually → numbering is wrong.

---

# 4) Label notation (symbols) + parsing/writing format

This is the most “annoying but small” change.

## 4a) Stage labels (what characters mean)

**What to look for**

- which characters represent empty: `s`, `.`, `0`, `_`, `-`, `empty`
- symbol set: digits `1..9`, `X/O`, `+ - =`, etc.
- ambiguous glyphs: `O` vs `0`, `S` vs `5`, `I` vs `1`

**What to change**

- `parse_kv_txt(...)`: map whatever they use into your internal canonical labels.
- `write_stage_files(...)`: write **exactly** the notation they demand.

**Minimal approach**

- Decide your internal convention, e.g. internal empty = `"s"`.
- In `parse_*`: convert any of `[".", "_", "empty"]` → `"s"`.
- In `write_*`: convert `"s"` back into whatever the statement asks.

## 4b) File format differences

**What to look for**

- keys are `1..N²` OR `(row col)` OR `"r,c"` OR they store 81 chars in one line (Sudoku-style)

**What to change**

- only the parser + writer:
  - `parse_*` reads the new format into `Dict[cell_id] = label`
  - `write_*` outputs in the required format

✅ You do **not** change warp/split/classify for this.

---

# 5) Preprocessing knobs (only if training accuracy is low)

**What to look for**

- lighting variations, grid lines bleeding into cells, faint symbols

**What to change (via `.env`)**

- `EX2_WARP_SIZE` (increase if symbols are small; Sudoku usually needs bigger)
- `EX2_CELL_MARGIN` (auto-tune is best)
- `EX2_BIN_INV` (flip if background/ink polarity differs)
- optionally switch Otsu → adaptive threshold (one small function edit)

**Minimal decision rule**

- If symbols are getting cut → decrease margin.
- If grid lines pollute symbol → increase margin slightly.
- If threshold fails under uneven light → adaptive threshold.

---

# The “rare but big” changes (when you’re basically forced to change logic)

These are the cases where you’re _not_ just adapting variables/parsers:

## A) Board is not a clean planar grid / cannot be warped reliably

Examples:

- curved paper, strong perspective + glare, board partially off-frame
- multiple boards in one image
- board edges not detectable (no strong contour)

**Then you need**

- explicit detection (e.g., detect corners/lines via Hough, or detect cells directly)
- or manual ROI/corner input (if allowed)

## B) Cells are not uniform / not axis-aligned after warp

Examples:

- irregular cell sizes, non-square cells, circular boards, hex grids

**Then you need**

- custom cell locator (grid line detection or template geometry), not just `split into equal chunks`

## C) Symbols are not “whole-cell classification”

Examples:

- multiple marks per cell, handwriting that overlaps grid lines
- need to detect _position_ inside cell (e.g., Sudoku pencil marks)
- requires object detection/segmentation rather than classification

**Then you need**

- segmentation or detection per cell (connected components, contour extraction, or a learned detector)

## D) Task requires post-logic beyond reading the board

Examples:

- “solve the Sudoku”
- “compute expression result”
- “determine winner + winning line”
- “validate board constraints”

This isn’t a CV rewrite, but you must add **a new logic layer** after recognition.

---

# Minimal “exam playbook” (10 minutes on a new similar task)

1. **Open one training label file** and answer:
   - grid size N?
   - indexing order?
   - symbol set + empty symbol?
   - file output format?

2. Update only:
   - `.env` paths
   - `EX2_GRID_N`
   - parser/writer to match format + notation
   - add numbering scheme candidates if unclear (row_major always included)

3. Run training auto-eval:
   - if exact-match improves sharply with a different scheme/margin → lock it
   - if still poor → tweak preprocessing (threshold + warp size)
   - if still poor and warped board looks wrong → geometry/detection problem (big-change category)

---

## The single most important thing to keep

Keep the **auto-search on training** for:

- numbering scheme
- `cell_inner_margin` (and optionally `warp_size`)

That is what turns “unknown but similar” exercises into “change 2–3 values and done”.

If you paste the **first 10 lines of the new training label format** (and tell me N), I can tell you exactly which of the 5 change-points apply and what mapping rules to implement—without touching the rest of your template.
