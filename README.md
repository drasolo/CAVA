# CAVA — Concepte și Aplicații în Vederea Artificială

Coursework, projects, and exam prep for the CAVA course (Univ. of Bucharest, FMI —
profs. Bogdan Alexe & Radu Ionescu). Organized **per module, in rough chronological order**.

> **Heavy files (images, zips, models, big PDFs) are intentionally NOT on GitHub** — they live in
> the full local mirror `D:\_QUARANTINE\2026-08-23\CAVA-ALL_BACKUP_2026-06-03`. On GitHub you get the code, notes,
> annotations, statements, and the complete folder structure (empty data folders are kept via
> `.gitkeep`). See `.gitignore` for exactly what's excluded.

## Timeline / layout

| Folder | What | When |
|--------|------|------|
| `00_course/`        | Lectures (`lectures/`, 2–3 sets), labs (`labs/`, sets A/B + tema2), notes | whole course |
| `01_tema2-2024/`    | Prior-year Tema 2 (self-contained: statement · data · code · evaluation) | Jan 2025 |
| `02_exam-jun2025/`  | The lab exam sat 6 Jun 2025: `subjects/`, `my-attempt/`, `reference_343_Gabroveanu/` | Jun 2025 |
| `03_tema1-qwirkle/` | Tema 1 — Qwirkle board scoring (statement · data · code · templates · docs · output · debug) | Nov–Dec 2025 |
| `04_tema2-scooby/`  | Tema 2 — Scooby-Doo face detection/recognition (statement · data · code · docs · models) | Feb 2026 |
| `05_exam-prep/`     | Upcoming-exam prep: `skeletons/`, `templates/`, `practice-subjects/`, `attempt_Lucrare1|2/` | Feb–Jun 2026 |
| `06_practice/`      | Practice system: 33 auto-checked drills, timed mock exams with sealed tests, theory quiz. Start with its README | Oct 2026 → |
| `tools/`            | Helper scripts (`structure_maker*.bat`, `invert_png.py`) + archived `structure*.txt` dumps | — |

> Naming note: `01_tema2-2024` (last year's homework) predates `03_tema1-qwirkle` (this year's),
> so the numbers follow the **calendar**, not the assignment numbers.

## Per-module convention

Each project folder is self-contained so a statement always sits next to its data and code:

```
<module>/
├─ statement/   the problem PDF(s)
├─ data/        antrenare · testare · validare
├─ code/        the solution
├─ docs/        notes, plans, roadmaps
└─ output/ debug/ evaluare/ models/ …   (as applicable)
```

## Recreate a fresh structure dump

```bat
tools\structure_maker.bat
```

## Restore the full data (from the local mirror)

Everything excluded here is in `D:\_QUARANTINE\2026-08-23\CAVA-ALL_BACKUP_2026-06-03` (a complete byte-for-byte copy
taken before the reorganization).
