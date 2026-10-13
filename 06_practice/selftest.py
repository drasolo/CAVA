"""Tests for the practice tooling itself (not the drills): python selftest.py

Runs against a throwaway me/ folder, so your real progress is never touched.
"""
import datetime as dt
import os
import random
import sys
import tempfile
from pathlib import Path

os.environ["CAVA_ME"] = tempfile.mkdtemp(prefix="cava-selftest-")
os.environ["CAVA_NO_WINDOW"] = "1"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import check  # noqa: E402
from cava_practice import concepts, flow, leitner, session  # noqa: E402
from cava_practice.testing import check as expect  # noqa: E402

D = dt.date


def smooth_pass(day, runs=1, right=True):
    return {"passed": day, "passed_at": D.fromisoformat(day).toordinal(), "runs_to_pass": runs,
            "prediction_right": right}


def test_phases():
    strict = flow.strict_stems()
    expect(strict[0] == "d00_gray" and len(strict) == 10 and strict[-1] == "d21_hough_lines",
           f"strict drills are the first 10: {strict}")
    expect(flow.phase("d00_gray", {}) == "strict", "d00 starts strict")
    expect(flow.phase("d22_ransac", {}) == "light", "drill 11 is light")
    two = {"d00_gray": smooth_pass("2026-10-13"), "d01_crop": smooth_pass("2026-10-14")}
    expect(not flow.early_exit(two), "two smooth drills are not enough")
    three = dict(two, d02_uint8=smooth_pass("2026-10-15"))
    expect(flow.early_exit(three) and flow.phase("d03_resize", three) == "light",
           "three smooth strict drills in a row end the strict phase")
    broken = dict(three, d01_crop=smooth_pass("2026-10-14", runs=5))
    expect(not flow.early_exit(broken), "a rough drill in the middle resets the run")
    wrong = dict(three, d02_uint8=smooth_pass("2026-10-15", right=False))
    expect(not flow.early_exit(wrong), "a wrong prediction isn't smooth")


def test_redos():
    exam = D(2027, 2, 1)
    s = flow.schedule_redos(D(2026, 10, 13), "strict", False, exam)
    expect(s == ["2026-10-14", "2026-10-20", "2027-01-25"], f"strict redos: {s}")
    s = flow.schedule_redos(D(2026, 10, 13), "light", False, exam)
    expect(s == ["2026-10-16", "2027-01-25"], f"light redos: {s}")
    s = flow.schedule_redos(D(2026, 10, 13), "light", True, exam)
    expect(s[0] == "2026-10-14", f"an assisted pass comes back after a day: {s}")
    s = flow.schedule_redos(D(2027, 1, 30), "strict", False, exam)
    expect(s == ["2027-01-31", "2027-02-06"], f"no pre-exam redo once it has passed: {s}")
    e = {"redos": ["2026-10-14", "2026-10-20"], "redo_log": []}
    expect(flow.due_redo(e, D(2026, 10, 13)) is None, "nothing due before the date")
    expect(flow.due_redo(e, D(2026, 10, 15)) == "2026-10-14", "due on and after the date")
    e["redo_log"].append({"due": "2026-10-14", "ok": True})
    expect(flow.due_redo(e, D(2026, 10, 15)) is None, "a good redo clears it")
    e = {"redos": ["2026-10-14", "2026-10-15"], "redo_log": [{"due": "2026-10-14", "ok": False}]}
    expect(flow.due_redo(e, D(2026, 10, 14)) is None, "a bad redo uses up its day too")
    expect(flow.due_redo(e, D(2026, 10, 15)) == "2026-10-15", "and the retry is due the next day")
    expect(flow.redo_went_well(1, 4, 30) and not flow.redo_went_well(2, 4, 30)
           and not flow.redo_went_well(1, 20, 30), "a redo counts with 1 run in a third of the time")


def test_hints():
    d00 = flow.hint_steps("d00_gray")
    expect(len(d00) == 2 and d00[0].startswith("In a BGR image"), f"d00 ladder: {d00}")
    d33 = flow.hint_steps("d33_symbol_cell")
    expect(len(d33) == 4 and d33[0].startswith("(pick one") and "1. Drop" in d33[0], f"d33 ladder: {d33}")
    expect(flow.hint_steps("d11_gaussian") == [], "drills without hints have an empty ladder")
    for stem in flow.stems():
        src = (ROOT / "drills" / f"{stem}.py").read_text(encoding="utf-8")
        expect("\nHINT   " not in src and "\nIDEAS" not in src, f"{stem} still shows its hints")
    e = {"opened_at": 1000.0}
    expect(flow.hint_unlocked_at(e, "strict") == 1000 + 15 * 60, "strict hints wait 15 minutes")
    expect(flow.hint_unlocked_at(e, "light") == 0, "light hints are immediate")


def test_stub():
    for stem in flow.stems():
        f = ROOT / "drills" / f"{stem}.py"
        src = f.read_text(encoding="utf-8")
        expect(check.stub(src) == src.rstrip() + "\n", f"stub({stem}) must equal the handed-out file")
    solved = '"""doc"""\nimport numpy as np\n\n\ndef f(x):\n    """keep me"""\n    y = x + 1\n    return y\n'
    expect(check.stub(solved) == '"""doc"""\nimport numpy as np\n\n\ndef f(x):\n    """keep me"""\n'
           '    raise NotImplementedError\n', "a solved function goes back to a stub, docstring kept")


def test_leitner():
    today = D(2026, 10, 13)
    stats = {}
    keys = ["a", "b", "c"]
    expect(leitner.pick(stats, keys, today, 2, random.Random(0)).__len__() == 2, "new items fill a session")
    leitner.answer(stats, "a", True, today)
    leitner.answer(stats, "b", False, today)
    expect(stats["a"]["box"] == 1 and stats["a"]["due"] == "2026-10-16", f"right -> box 1, 3 days: {stats['a']}")
    expect(stats["b"]["box"] == 0 and stats["b"]["due"] == "2026-10-14", f"wrong -> box 0, tomorrow: {stats['b']}")
    expect(leitner.due_keys(stats, keys, D(2026, 10, 14)) == ["b"], "only b is due tomorrow")
    picked = leitner.pick(stats, keys, D(2026, 10, 14), 2, random.Random(0))
    expect(picked == ["b", "c"], f"due first, then new: {picked}")
    for _ in range(10):
        leitner.answer(stats, "a", True, today)
    expect(stats["a"]["box"] == 4, "the box tops out at 4 (a month)")
    old = leitner.migrate({"q": {"seen": 4, "right": 1}, "r": {"seen": 5, "right": 5}})
    expect(old["q"]["box"] == 0 and old["r"]["box"] == 2, f"old stats get a box from their hit rate: {old}")


def test_clock():
    s = {"last_tick": 0, "active_seconds": 0}
    session.tick(s, 60, last_activity=0, idle_seconds=600)
    expect(s["active_seconds"] == 60 and not s["idle"], "active time counts")
    session.tick(s, 900, last_activity=0, idle_seconds=600)
    expect(s["active_seconds"] == 600 and s["idle"], f"time past the idle limit doesn't: {s}")
    session.tick(s, 1000, last_activity=950, idle_seconds=600)
    expect(s["active_seconds"] == 700, f"activity resumes the clock: {s}")
    rows = [{"date": "2026-10-12", "kind": "drill", "minutes": 30, "counted": True},
            {"date": "2026-10-14", "kind": "review", "minutes": 25, "counted": True},
            {"date": "2026-10-14", "kind": "drill", "minutes": 10, "counted": False},
            {"date": "2026-10-19", "kind": "drill", "minutes": 40, "counted": True}]
    w = session.week_summary(rows, D(2026, 10, 15), live_minutes=20, live_kind="tema")
    expect(w["minutes"] == 75 and w["by_kind"] == {"drill": 30, "review": 25, "tema": 20},
           f"the week runs Monday to Sunday, counted sessions plus the live one: {w}")
    expect(session.fmt(160) == "2h40" and session.fmt(45) == "45m", "time formatting")


def test_concepts():
    g = concepts.graph()
    expect(not concepts.missing(g), f"every prerequisite exists: {concepts.missing(g)}")
    expect(concepts.cycle(g) is None, f"no concept needs itself: {concepts.cycle(g)}")
    for stem in flow.strict_stems():
        expect(stem in g["drills"], f"strict drill {stem} has a concept map")
    for cid, c in g["concepts"].items():
        expect(c.get("explain") and c.get("check", {}).get("q") and c["check"].get("a"),
               f"{cid} needs an explanation and a check question with an answer")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
