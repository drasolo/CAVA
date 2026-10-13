"""Run the practice drills.

    python check.py              list every drill and your progress
    python check.py d12          test your drills/d12_*.py
    python check.py d12 --ref    test the reference solution (proves the bar is reachable)
    python check.py next         test the first drill you haven't passed yet
    python check.py --all --ref  test every reference solution
"""
import argparse
import datetime as dt
import importlib.util
import json
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
PROGRESS = ROOT / "progress.json"


class Fail(AssertionError):
    pass


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


def read_progress():
    try:
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_progress(p):
    PROGRESS.write_text(json.dumps(p, indent=2, sort_keys=True), encoding="utf-8")


def run_one(stem: str, ref: bool) -> bool:
    folder = "solutions" if ref else "drills"
    target = ROOT / folder / f"{stem}.py"
    test_file = ROOT / "tests" / f"{stem}.py"
    print(f"\n=== {stem}  ({'reference' if ref else 'your code'})")
    print(f"    {title(ROOT / 'drills' / f'{stem}.py')}")
    start = time.perf_counter()
    try:
        mod = load(target, f"{folder}_{stem}")
        tests = load(test_file, f"tests_{stem}")
        tests.test(mod)
    except NotImplementedError as e:
        where = traceback.extract_tb(e.__traceback__)[-1]
        print(f"    TODO  {where.name}() is not implemented yet{': ' + str(e) if str(e) else ''}")
        return False
    except AssertionError as e:
        print(f"    FAIL  {e}")
        return False
    except Exception:
        print("    ERROR your code crashed:")
        print("    " + traceback.format_exc().replace("\n", "\n    "))
        return False
    print(f"    PASS  ({time.perf_counter() - start:.1f}s)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drill", nargs="?", help="drill id prefix (d12), 'next', or nothing to list")
    ap.add_argument("--ref", action="store_true", help="test the reference solution instead")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    stems = [p.stem for p in drills()]
    progress = read_progress()

    if args.all:
        results = {s: run_one(s, args.ref) for s in stems}
        bad = [s for s, ok in results.items() if not ok]
        print(f"\n{len(stems) - len(bad)}/{len(stems)} passed" + (f"; failing: {', '.join(bad)}" if bad else ""))
        sys.exit(1 if bad else 0)

    if not args.drill:
        level = None
        for s in stems:
            if s[1] != level:
                level = s[1]
                print(f"\nLevel {level}")
            mark = "x" if progress.get(s, {}).get("passed") else " "
            print(f"  [{mark}] {s:<24} {title(ROOT / 'drills' / f'{s}.py')}")
        done = sum(1 for s in stems if progress.get(s, {}).get("passed"))
        print(f"\n{done}/{len(stems)} drills passed")
        return

    if args.drill == "next":
        todo = [s for s in stems if not progress.get(s, {}).get("passed")]
        if not todo:
            print("Every drill passed. Move on to the mocks (see README).")
            return
        matches = todo[:1]
    else:
        matches = [s for s in stems if s.startswith(args.drill)]
    if not matches:
        sys.exit(f"No drill matches '{args.drill}'. Run `python check.py` to list them.")

    for stem in matches:
        ok = run_one(stem, args.ref)
        if not args.ref:
            entry = progress.setdefault(stem, {"attempts": 0})
            entry["attempts"] += 1
            if ok and not entry.get("passed"):
                entry["passed"] = dt.date.today().isoformat()
            write_progress(progress)


if __name__ == "__main__":
    main()
