"""Grade a mock exam attempt with the statement's rules.

    python mocks/grade.py mocks/attempts/2025-06-06_20261012-1530

Ex1: accuracy of outputs/ex1_predictions.txt on the official TEST split.
Ex2: outputs/ex2/* against the sealed answers, all-or-nothing per image.
Ex2 points are projected to the full exam (your fraction correct x task points).
"""
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from exams import EXAMS_SPEC  # noqa: E402
from cava_practice import exam_scoring as es  # noqa: E402
from cava_practice.faces import SPECS, split  # noqa: E402


def grade_ex1(att, exam):
    pts = SPECS[exam]["points"]
    pred_file = att / "outputs" / "ex1_predictions.txt"
    test = {it.name: it.label for it in split(exam) if it.part == "test"}
    if not pred_file.exists():
        print(f"Ex1: no {pred_file.relative_to(att)}  -> 0 / {pts}")
        return 0.0
    preds = {}
    for line in pred_file.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 2:
            preds[p[0]] = p[1]
    missing = [n for n in test if n not in preds]
    correct = sum(preds.get(n) == lab for n, lab in test.items())
    acc = correct / len(test)
    score = acc * pts
    print(f"Ex1: accuracy {correct}/{len(test)} = {acc:.1%}  -> {score:.2f} / {pts}")
    if missing:
        print(f"     {len(missing)} test images have no prediction (counted wrong), e.g. {missing[:3]}")
    extra = set(preds) - set(test)
    if extra and len(extra) > len(test) // 2:
        print(f"     note: {len(extra)} predictions are for non-test images (ignored). Is your split right?")
    return score


def grade_ex2(att, exam):
    spec = EXAMS_SPEC[exam]["ex2"]
    kind, tp = spec["kind"], spec["task_points"]
    sealed = att / ".sealed"
    out = att / "outputs" / "ex2"
    imgs = sorted(p.stem for p in (att / "data" / "ex2" / "testare").iterdir())
    oks = [[] for _ in tp]
    for stem in imgs:
        if kind == "xo":
            r = es.score_xo(out / f"{stem}.txt", sealed / f"{stem}.txt") if (out / f"{stem}.txt").exists() else (False, False, "no output file")
            flags, note = r[:2], r[2]
        elif kind == "single":
            r = es.score_single(out / f"{stem}.txt", sealed / f"{stem}.txt") if (out / f"{stem}.txt").exists() else (False, "no output file")
            flags, note = r[:1], r[1]
        else:
            files = [out / f"{stem}_1.txt", out / f"{stem}_2.txt"]
            if all(f.exists() for f in files):
                r = es.score_ring(*files, sealed / f"{stem}_1.txt", sealed / f"{stem}_2.txt")
            else:
                r = (False, False, "missing _1/_2 output file")
            flags, note = r[:2], r[2]
        for i, f in enumerate(flags):
            oks[i].append(bool(f))
        marks = " ".join("ok " if f else "-- " for f in flags)
        print(f"     {stem}: {marks} {note}")
    total = 0.0
    for i, (task_ok, pts) in enumerate(zip(oks, tp)):
        frac = sum(task_ok) / len(task_ok)
        total += frac * pts
        print(f"Ex2 task {i + 1}: {sum(task_ok)}/{len(task_ok)} images  -> {frac * pts:.2f} / {pts} (projected)")
    return total


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    att = Path(sys.argv[1]).resolve()
    meta = json.loads((att / "attempt.json").read_text())
    exam = meta["exam"]
    started = dt.datetime.fromisoformat(meta["started"])
    elapsed = dt.datetime.now() - started
    print(f"Mock {exam}: {EXAMS_SPEC[exam]['title']}")
    h, m = divmod(int(elapsed.total_seconds()) // 60, 60)
    print(f"Time since start: {h}h{m:02d}m" + ("   (over the 2h limit)" if elapsed > dt.timedelta(hours=2) else ""))
    print()
    s1 = grade_ex1(att, exam)
    print("Ex2 per image:")
    s2 = grade_ex2(att, exam)
    total = 1 + s1 + s2
    print(f"\nTOTAL: {total:.2f} / 10  (1 point ex officio)")
    meta.setdefault("gradings", []).append({"at": dt.datetime.now().isoformat(timespec="seconds"),
                                            "ex1": round(s1, 3), "ex2": round(s2, 3), "total": round(total, 3),
                                            "minutes": int(elapsed.total_seconds() // 60)})
    (att / "attempt.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
