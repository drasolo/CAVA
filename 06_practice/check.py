"""Run the practice drills, inside the learning flow.

    python check.py              list every drill, your progress and due redos
    python check.py d12          the lesson for d12 (read mode), or your code (--write / no lesson)
    python check.py d12 --hint   show the next hint step (strict drills: after 15 minutes)
    python check.py d12 --redo   start a from-memory redo: your code is archived, the stub comes back
    python check.py next         a due redo if there is one, else the first drill not passed
    python check.py d12 --ref    test the reference solution (proves the bar is reachable)
    python check.py --all --ref  test every reference solution

The flow (predictions, notes, cards, redos) is described in cava_practice/flow.py.
Set CAVA_FLOW=off to run a drill with no questions asked.
"""
import argparse
import ast
import datetime as dt
import importlib.util
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

# Library chatter buries the PASS/FAIL lines. joblib shells out to `wmic`
# (gone on Windows 11) to count physical cores unless LOKY_MAX_CPU_COUNT is
# set BELOW the logical core count.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", module="sklearn")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cava_practice import flow, journal, state  # noqa: E402


def has_lesson(stem: str) -> bool:
    return (ROOT / "lessons" / f"{stem}.yaml").exists()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def drills():
    return sorted(p for p in (ROOT / "drills").glob("d[0-9][0-9]_*.py"))


def title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().strip('"').strip()
        if line:
            return line
    return path.stem


def statement(stem: str) -> str:
    """The drill's docstring: course link, task, rules. The hints live in hints/."""
    return ast.get_docstring(ast.parse((ROOT / "drills" / f"{stem}.py").read_text(encoding="utf-8"))) or ""


def stub(source: str) -> str:
    """The drill file as it was handed out: same docstring and signatures, empty bodies."""
    tree = ast.parse(source)
    lines = source.split("\n")
    # Replace each top-level function body, bottom to top so line numbers stay valid.
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for fn in reversed(funcs):
        first = fn.body[0]
        keep_doc = isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
            and isinstance(first.value.value, str)
        start = (fn.body[1].lineno if keep_doc and len(fn.body) > 1 else
                 (first.end_lineno + 1 if keep_doc else first.lineno)) - 1
        end = fn.end_lineno
        indent = " " * (fn.col_offset + 4)
        lines[start:end] = [indent + "raise NotImplementedError"]
    return "\n".join(lines).rstrip() + "\n"


def run_one(stem: str, ref: bool, quiet: bool = False):
    """Run a drill's tests. Returns (ok, message)."""
    folder = "solutions" if ref else "drills"
    target = ROOT / folder / f"{stem}.py"
    test_file = ROOT / "tests" / f"{stem}.py"
    if not quiet:
        print(f"\n=== {stem}  ({'reference' if ref else 'your code'})")
        print(f"    {title(ROOT / 'drills' / f'{stem}.py')}")
    start = time.perf_counter()
    try:
        mod = load(target, f"{folder}_{stem}")
        tests = load(test_file, f"tests_{stem}")
        tests.test(mod)
    except NotImplementedError as e:
        where = traceback.extract_tb(e.__traceback__)[-1]
        msg = f"TODO  {where.name}() is not implemented yet{': ' + str(e) if str(e) else ''}"
        print(f"    {msg}")
        return False, msg
    except AssertionError as e:
        print(f"    FAIL  {e}")
        return False, f"FAIL  {e}"
    except Exception as e:
        print("    ERROR your code crashed:")
        print("    " + traceback.format_exc().replace("\n", "\n    "))
        return False, f"ERROR {type(e).__name__}: {e}"
    print(f"    PASS  ({time.perf_counter() - start:.1f}s)")
    return True, "PASS"


# ── the flow around one run ───────────────────────────────────────────

def flow_on() -> bool:
    return os.environ.get("CAVA_FLOW", "on") != "off"


def save_card(stem: str, card: dict) -> None:
    cards = state.load(state.CARDS)
    cards[stem] = {"q": card["q"], "a": card["a"], "topic": f"card {stem[:3]}"}
    state.save(state.CARDS, cards)


def attempt(stem: str) -> bool:
    """One checker run with everything the flow asks around it. Returns True on PASS."""
    progress = state.load(state.PROGRESS)
    entry = progress.setdefault(stem, {})
    first_time = "opened_at" not in entry
    if first_time:
        entry["opened_at"] = state.now()
        print(f"\n{statement(stem)}\n")
    ph = flow.phase(stem, progress) if not entry.get("passed") else entry.get("phase", "light")
    redo = entry.get("redo_active")

    if flow_on() and not entry.get("passed") and not redo:
        entry.setdefault("phase", ph)
        flow.predict(entry, ph)
        state.save(state.PROGRESS, progress)
        if flow.needs_failure_note(entry, ph):
            flow.failure_note(entry)
            state.save(state.PROGRESS, progress)
        expected = flow.expect(entry, ph)
    else:
        expected = ""

    ok, msg = run_one(stem, ref=False)
    state.touch()
    entry["attempts"] = entry.get("attempts", 0) + 1

    if redo:
        redo["runs"] = redo.get("runs", 0) + 1
        if ok:
            minutes = (state.now() - redo["started_at"]) / 60
            good = flow.redo_went_well(redo["runs"], minutes, entry.get("minutes", 0))
            entry.setdefault("redo_log", []).append(
                {"due": redo["due"], "ok": good, "minutes": round(minutes, 1), "runs": redo["runs"],
                 "on": state.today().isoformat()})
            if not good:
                again = (state.today() + dt.timedelta(days=1)).isoformat()
                entry["redos"] = sorted(set(entry.get("redos", []) + [again]))
            best = min((r["minutes"] for r in entry["redo_log"] if r["ok"]), default=None)
            print(f"\n    Redo done in {minutes:.1f} min, {redo['runs']} run(s): "
                  + ("it stuck." if good else "it comes back tomorrow (needs 1 run and a third of the first time).")
                  + (f" Best: {best:.1f} min." if best else ""))
            del entry["redo_active"]
    else:
        entry.setdefault("run_log", []).append(
            {"at": state.now(), "ok": ok, "msg": msg[:300], "expect": expected})
        if ok and not entry.get("passed"):
            if flow_on():
                flow.on_pass(entry, ph)
            entry["passed"] = state.today().isoformat()
            entry["passed_at"] = state.now()
            entry["runs_to_pass"] = len(entry["run_log"])
            entry["minutes"] = round((state.now() - entry["opened_at"]) / 60, 1)
            entry["redos"] = flow.schedule_redos(state.today(), ph, bool(entry.get("assisted")),
                                                 dt.date.fromisoformat(state.config()["exam"]))
            if entry.get("card"):
                save_card(stem, entry["card"])
            print(f"\n    Recorded. First from-memory redo: {entry['redos'][0]}.")
            if ph == "strict" and flow.early_exit(progress):
                print("    Three smooth strict drills in a row: the strict phase is over. Nice.")
    state.save(state.PROGRESS, progress)
    journal.write()
    return ok


def read_mode(args=None) -> bool:
    if args is not None and getattr(args, "write", False):
        return False
    return state.config().get("mode", "read") == "read"


def lesson(stem: str, redo: bool = False) -> bool:
    """Read mode: run the drill's lesson and record it like a pass (or a redo)."""
    from cava_practice import read
    progress = state.load(state.PROGRESS)
    entry = progress.get(stem, {})
    redo = redo or bool(entry.get("passed"))
    due = flow.due_redo(entry, state.today()) if entry.get("passed") else None
    result = read.run_lesson(stem, redo=redo)
    if result is None:
        return False
    progress, entry, rec = result
    pts, total = rec["score"]
    if not entry.get("passed"):
        ph = flow.phase(stem, progress)
        entry.update({"phase": ph, "passed": state.today().isoformat(), "passed_at": state.now(),
                      "minutes": rec["minutes"], "prediction_right": rec.get("predict"),
                      "runs_to_pass": 1 if pts >= total - 1 else 3})
        entry["redos"] = flow.schedule_redos(state.today(), ph, bool(entry.get("assisted")),
                                             dt.date.fromisoformat(state.config()["exam"]))
        if entry.get("card"):
            save_card(stem, entry["card"])
        print(f"  Recorded. It comes back for a redo on {entry['redos'][0]}.")
        if ph == "strict" and flow.early_exit(progress):
            print("  Three smooth lessons in a row: the strict phase is over. Nice.")
    else:
        # One slip is fine; more and it comes back tomorrow.
        good = pts >= total - 1
        entry.setdefault("redo_log", []).append(
            {"due": due or state.today().isoformat(), "ok": good, "minutes": rec["minutes"],
             "runs": total - pts + 1, "on": state.today().isoformat()})
        if not good:
            entry["redos"] = sorted(set(entry.get("redos", []) + [(state.today() + dt.timedelta(days=1)).isoformat()]))
        print("  It stuck." if good else "  A few slips: it comes back tomorrow.")
    progress[stem] = entry
    state.save(state.PROGRESS, progress)
    journal.write()
    state.touch()
    return True


def hint(stem: str) -> None:
    progress = state.load(state.PROGRESS)
    entry = progress.setdefault(stem, {})
    entry.setdefault("opened_at", state.now())
    ph = entry.get("phase") or flow.phase(stem, progress)
    steps = flow.hint_steps(stem)
    shown = len(entry.get("hints", []))
    if not steps:
        print("    This drill has no written hints. Try `python coach.py` -> 'I don't get the concept'.")
    elif shown >= len(steps):
        print("    No more hints. Stuck past ~45 minutes? Read solutions/, close it, redo it tomorrow.")
        for s in steps:
            print(f"    - {s}")
    else:
        wait = flow.hint_unlocked_at(entry, ph) - state.now()
        if wait > 0:
            print(f"    Hints unlock in {wait / 60:.0f} min. Stay with it: being stuck is where it sticks.")
        else:
            entry.setdefault("hints", []).append(state.now())
            for k, s in enumerate(steps[:shown + 1], 1):
                print(f"    hint {k}/{len(steps)}: {s}")
    state.save(state.PROGRESS, progress)


def start_redo(stem: str) -> None:
    progress = state.load(state.PROGRESS)
    entry = progress.get(stem, {})
    if not entry.get("passed"):
        sys.exit(f"{stem} isn't passed yet: there is nothing to redo.")
    f = ROOT / "drills" / f"{stem}.py"
    archive = state.path("archive")
    archive.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    (archive / f"{stem}-{stamp}.py").write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    f.write_text(stub(f.read_text(encoding="utf-8")), encoding="utf-8")
    entry["redo_active"] = {"due": flow.due_redo(entry, state.today()) or state.today().isoformat(),
                            "started_at": state.now(), "runs": 0}
    progress[stem] = entry
    state.save(state.PROGRESS, progress)
    print(f"    Your code is in me/archive/{stem}-{stamp}.py. {f.name} is blank again.")
    print(f"    Clock started. Write it from memory, then `python check.py {stem[:3]}`.")


def listing(stems, progress) -> None:
    today = state.today()
    level = None
    for s in stems:
        if s[1] != level:
            level = s[1]
            print(f"\nLevel {level}")
        e = progress.get(s, {})
        mark = "~" if e.get("passed") and flow.due_redo(e, today) else ("x" if e.get("passed") else " ")
        tag = "  (strict)" if s in flow.strict_stems() and not e.get("passed") and flow.phase(s, progress) == "strict" else ""
        tag += "" if has_lesson(s) else "  (no lesson yet)"
        print(f"  [{mark}] {s:<24} {title(ROOT / 'drills' / f'{s}.py')}{tag}")
    done = sum(1 for s in stems if progress.get(s, {}).get("passed"))
    due = [s for s in stems if progress.get(s, {}).get("passed") and flow.due_redo(progress[s], today)]
    print(f"\n{done}/{len(stems)} drills passed" + (f" · redo due: {', '.join(due)}  ([~])" if due else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drill", nargs="?", help="drill id prefix (d12), 'next', or nothing to list")
    ap.add_argument("--ref", action="store_true", help="test the reference solution instead")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--hint", action="store_true", help="show the next hint step")
    ap.add_argument("--redo", action="store_true", help="start a from-memory redo")
    ap.add_argument("--write", action="store_true", help="write the code yourself instead of the lesson")
    args = ap.parse_args()

    stems = [p.stem for p in drills()]
    progress = state.load(state.PROGRESS)

    if args.all:
        results = {s: run_one(s, args.ref)[0] for s in stems}
        bad = [s for s, ok in results.items() if not ok]
        print(f"\n{len(stems) - len(bad)}/{len(stems)} passed" + (f"; failing: {', '.join(bad)}" if bad else ""))
        sys.exit(1 if bad else 0)

    if not args.drill:
        listing(stems, progress)
        return

    if args.drill == "next":
        today = state.today()
        due = [s for s in stems if progress.get(s, {}).get("passed") and flow.due_redo(progress[s], today)
               and not progress[s].get("redo_active")]
        if due:
            if read_mode(args) and has_lesson(due[0]):
                print(f"Redo due: {due[0]}.")
                lesson(due[0], redo=True)
            else:
                print(f"Redo due: {due[0]}. Start it with `python check.py {due[0][:3]} --redo` (your code gets archived).")
            return
        active = [s for s in stems if progress.get(s, {}).get("redo_active")]
        todo = active or [s for s in stems if not progress.get(s, {}).get("passed")]
        if not todo:
            print("Every drill passed and no redo is due. Move on to the mocks (see README).")
            return
        matches = todo[:1]
    else:
        matches = [s for s in stems if s.startswith(args.drill)]
    if not matches:
        sys.exit(f"No drill matches '{args.drill}'. Run `python check.py` to list them.")

    for stem in matches:
        if args.ref:
            run_one(stem, ref=True)
        elif args.hint:
            hint(stem)
        elif read_mode(args) and has_lesson(stem):
            lesson(stem, redo=args.redo)
        elif args.redo:
            start_redo(stem)
        else:
            if read_mode(args):
                print(f"    {stem} has no lesson yet, so this is the write-it-yourself flow.")
            attempt(stem)


if __name__ == "__main__":
    main()
