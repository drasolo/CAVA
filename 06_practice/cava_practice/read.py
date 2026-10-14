"""Read mode: learn a drill from its working code instead of writing it.

A lesson (lessons/<stem>.yaml) walks eight steps:
  1 concept    the concept-map walk for the drill
  2 predict    one number from a tiny input, checked against the real code
  3 read       the solution, explained line by line ("?N" asks Claude about line N)
  4 order      the lines of one function shuffled: put them back in order
  5 fill       one key line blanked: pick it from four options
  6 bug        the code with one classic bug planted: find the line, say why
  7 tweak      change a parameter, predict the effect, then see it
  8 explain    two sentences and a card

Every answer that is code is checked by running the drill's real tests on it,
so a different-but-correct order or option also counts.
"""
import ast
import contextlib
import os
import importlib.util
import io
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import concepts, flow, paths, state, synth, tutor

PRACTICE = Path(__file__).resolve().parent.parent
LESSONS = PRACTICE / "lessons"
SOLUTIONS = PRACTICE / "solutions"
TESTS = PRACTICE / "tests"
STEPS = ["concept", "predict", "read", "order", "fill", "bug", "tweak", "explain"]


# ── loading and running code ─────────────────────────────────────────

def lesson(stem: str):
    f = LESSONS / f"{stem}.yaml"
    return yaml.safe_load(f.read_text(encoding="utf-8")) if f.exists() else None


def has_lesson(stem: str) -> bool:
    return (LESSONS / f"{stem}.yaml").exists()


def solution_source(stem: str) -> str:
    return (SOLUTIONS / f"{stem}.py").read_text(encoding="utf-8")


def load_source(source: str, name: str):
    """Import a module from source text (a temp file, so tracebacks and inspect work)."""
    tmp = Path(tempfile.mkdtemp(prefix="cava-read-")) / f"{name}.py"
    tmp.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_tests(stem: str, source: str):
    """Run the drill's tests on `source`. Returns (passed, message)."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            mod = load_source(source, f"read_{stem}")
            spec = importlib.util.spec_from_file_location(f"tests_{stem}", TESTS / f"{stem}.py")
            tests = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tests)
            tests.test(mod)
    except AssertionError as e:
        return False, str(e)
    except Exception as e:  # a crash is a failure too
        return False, f"{type(e).__name__}: {e}"
    return True, "PASS"


def replace_line(source: str, old: str, new: str) -> str:
    """Swap the one line whose stripped text is `old`, keeping its indentation."""
    lines = source.split("\n")
    hits = [i for i, l in enumerate(lines) if l.strip() == old.strip()]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one line '{old}', found {len(hits)}")
    i = hits[0]
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    lines[i] = indent + new.strip()
    return "\n".join(lines)


def function_body(source: str, func: str):
    """(start, end, indent, body lines) of a top-level function's body."""
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func)
    lines = source.split("\n")
    start, end = fn.body[0].lineno - 1, fn.end_lineno
    return start, end, " " * (fn.col_offset + 4), lines[start:end]


def with_body(source: str, func: str, body: list) -> str:
    start, end, indent, _ = function_body(source, func)
    lines = source.split("\n")
    lines[start:end] = [indent + b for b in body]   # a line's own leading spaces nest it (loops)
    return "\n".join(lines)


def probe_value(stem: str, probe: dict):
    """What the real code answers for the probe question."""
    mod = load_source(solution_source(stem), f"probe_{stem}")
    ns = {"np": np, "cv2": cv2, "synth": synth, "m": mod}
    ns.update({k: getattr(mod, k) for k in dir(mod) if not k.startswith("_")})
    exec(probe.get("setup", ""), ns)
    return eval(probe["ask"], ns)


def demo_image():
    """A real Qwirkle photo when it's on disk, else a synthetic board."""
    for p in sorted(paths.QWIRKLE_TRAIN.glob("*.jpg"))[:1]:
        img = cv2.imread(str(p))
        if img is not None:
            h, w = img.shape[:2]
            s = 480 / max(h, w)
            return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    scene = synth.board_scene(seed=1)
    return scene[0] if isinstance(scene, tuple) else scene   # board_scene also returns the corners


# ── talking to you ───────────────────────────────────────────────────

def say(text: str, pad: str = "    ") -> None:
    for line in str(text).strip().splitlines():
        print(pad + line)


def step(n: int, title: str) -> None:
    print(f"\n  ── {n}/8 · {title} " + "─" * max(0, 50 - len(title)))


def choose(options: list, rng) -> tuple:
    """Shuffle options; return (shuffled, letter of each original index)."""
    order = list(range(len(options)))
    rng.shuffle(order)
    letters = "abcdefgh"
    for k, i in enumerate(order):
        say(f"{letters[k]})  {options[i]}")
    return order, letters


def numbered(source_lines: list, mark=None) -> None:
    for k, line in enumerate(source_lines, 1):
        print(f"    {'>' if mark == k else ' '}{k:3d}  {line}")


# ── the eight steps ──────────────────────────────────────────────────

def do_concept(stem, L, rec, rng):
    step(1, "the concept first")
    if flow.ask("    Walk the ideas behind this drill? [Y/n] ", False).lower() == "n":
        rec["concept"] = "skipped"
        return
    concepts.descend(stem)
    rec["concept"] = "done"


def do_predict(stem, L, rec, rng):
    p = L.get("probe")
    step(2, "predict")
    if not p:
        rec["predict"] = None
        return
    say(p["question"])
    truth = probe_value(stem, p)
    guess = flow.ask("    Your number: ", True)
    try:
        ok = abs(float(guess) - float(truth)) <= p.get("tol", 0.5)
    except ValueError:
        ok = str(guess).strip().lower() == str(truth).strip().lower()
    say(f"The code says: {truth}. " + ("You had it." if ok else "Not quite: keep this in mind while you read."))
    if p.get("why"):
        say(p["why"])
    rec["predict"] = ok


def do_read(stem, L, rec, rng):
    step(3, "read the code, line by line")
    src = solution_source(stem).rstrip().split("\n")
    notes = {k.strip(): v for k, v in (L.get("lines") or {}).items()}
    for k, line in enumerate(src, 1):
        print(f"    {k:3d}  {line}")
        note = notes.get(line.strip())
        if note:
            for n in str(note).strip().splitlines():
                print(f"         # {n}")
    say("\nAny line unclear? Type its number for another explanation (Enter to go on).", "")
    asked = []
    while True:
        a = flow.ask("    line: ", False)
        if not a:
            break
        if a.isdigit() and 1 <= int(a) <= len(src):
            k = int(a)
            asked.append(k)
            print("    asking Claude ...")
            reply = tutor.ask(f"Explain this line of Python, from a computer vision drill, to a student who reads "
                              f"code better than they write it. The whole file:\n\n{chr(10).join(src)}\n\n"
                              f"The line: {src[k - 1].strip()}\n\nWhat it does, why it is there, and what would break "
                              "without it. Plain words.")
            say(reply, "      ")
    rec["read"] = {"asked": asked}


def do_order(stem, L, rec, rng):
    spec = L.get("order")
    step(4, "put the lines in order")
    if not spec:
        rec["order"] = None
        return
    func = spec["func"]
    src = solution_source(stem)
    body = spec.get("lines") or [l.strip() for l in function_body(src, func)[3] if l.strip()]
    body = [b.rstrip() for b in body]
    shuffled = body[:]
    while len(shuffled) > 1 and shuffled == body:
        rng.shuffle(shuffled)
    say(f"These are the lines of {func}(), shuffled. Type the order, e.g. 3 1 2"
        + (" (indentation shows what's inside a loop)." if any(b.startswith(" ") for b in body) else "."))
    for k, line in enumerate(shuffled, 1):
        print(f"    {k}.  {line}")
    tries = 0
    while True:
        tries += 1
        a = flow.ask("    order: ", True).replace(",", " ").split()
        try:
            idx = [int(x) - 1 for x in a]
            assert sorted(idx) == list(range(len(shuffled)))
        except (ValueError, AssertionError):
            say(f"Use each number 1 to {len(shuffled)} once.")
            continue
        ok, msg = run_tests(stem, with_body(src, func, [shuffled[i] for i in idx]))
        if ok:
            say("Correct: the tests pass with your order." + (" First try." if tries == 1 else ""))
            break
        say(f"Not yet. The tests say: {msg[:200]}")
        if tries >= 3 and flow.ask("    Show the right order? [y/N] ", False).lower().startswith("y"):
            say("\n".join(body))
            break
    rec["order"] = {"tries": tries}


def do_fill(stem, L, rec, rng):
    items = L.get("fill") or []
    step(5, "fill the gap")
    if not items:
        rec["fill"] = None
        return
    item = rng.choice(items)
    src = solution_source(stem)
    lines = src.rstrip().split("\n")
    k = next(i for i, l in enumerate(lines) if l.strip() == item["line"].strip())
    shown = lines[:]
    shown[k] = lines[k][:len(lines[k]) - len(lines[k].lstrip())] + "______"
    numbered(shown, mark=k + 1)
    say("\nWhich line goes in the gap?", "")
    options = [item["line"]] + item["wrong"]
    order, letters = choose(options, rng)
    a = flow.ask("    your pick: ", True).lower()[:1]
    pick = order[letters.index(a)] if a in letters[:len(options)] else -1
    ok = pick == 0
    if ok:
        say("Right.")
    else:
        if pick > 0:
            _, msg = run_tests(stem, replace_line(src, item["line"], options[pick]))
            say(f"With that line the tests say: {msg[:200]}")
        say(f"The line is: {item['line'].strip()}")
    say(item.get("why", ""))
    rec["fill"] = ok


def do_bug(stem, L, rec, rng):
    bugs = L.get("bugs") or []
    step(6, "find the bug")
    if not bugs:
        rec["bug"] = None
        return
    bug = rng.choice(bugs)
    src = solution_source(stem)
    broken = replace_line(src, bug["line"], bug["bad"])
    lines = broken.rstrip().split("\n")
    target = next(i for i, l in enumerate(lines) if l.strip() == bug["bad"].strip()) + 1
    say("One line in this code has a classic bug. The tests fail on it.")
    numbered(lines)
    a = flow.ask("    Which line number? ", True)
    found = a.strip() == str(target)
    say("Yes, that's the one." if found else f"It's line {target}: {bug['bad'].strip()}")
    why = flow.ask("    Why is it wrong? (one line) ", True)
    _, msg = run_tests(stem, broken)
    say(f"What the tests say about it: {msg[:200]}")
    say(f"The fix: {bug['line'].strip()}")
    say(bug.get("why", ""))
    rec["bug"] = {"found": found, "why": why}


def show(title: str, img) -> None:
    """Show images in a window; any key closes it."""
    if os.environ.get("CAVA_NO_WINDOW") == "1":
        print(f"    [image] {title}: {img.shape} {img.dtype}")
        return
    shown = img
    if shown.dtype != np.uint8:
        shown = cv2.normalize(shown.astype(np.float64), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow(title, shown)


def do_tweak(stem, L, rec, rng):
    t = L.get("tweak")
    step(7, "change a parameter, predict the effect")
    if not t:
        rec["tweak"] = None
        return
    say(t["question"])
    order, letters = choose(t["options"], rng)
    a = flow.ask("    your prediction: ", True).lower()[:1]
    pick = order[letters.index(a)] if a in letters[:len(t["options"])] else -1
    ok = pick == t["answer"]
    say(("Right. " if ok else f"The answer: {t['options'][t['answer']]}. ") + t.get("why", ""))
    if t.get("run"):
        mod = load_source(solution_source(stem), f"tweak_{stem}")
        ns = {"np": np, "cv2": cv2, "synth": synth, "m": mod, "show": show, "say": say,
              "demo_image": demo_image}
        ns.update({k: getattr(mod, k) for k in dir(mod) if not k.startswith("_")})
        exec(t["run"], ns)
        if t.get("images") and os.environ.get("CAVA_NO_WINDOW") != "1":
            say("(image windows open: press any key in one of them to close them)")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    rec["tweak"] = ok


def do_explain(stem, L, rec, rng, entry):
    step(8, "explain it, make a card")
    say("Two sentences: what it does, and why it matters later.")
    one = flow.ask("    1. ", True)
    two = flow.ask("    2. ", True)
    entry["explanation"] = f"{one} {two}".strip()
    q = flow.ask("    Card question (the trap or idea to remember): ", True)
    a = flow.ask("    Card answer: ", True)
    entry["card"] = {"q": q, "a": a}
    rec["explain"] = True


def score(rec: dict) -> tuple:
    """(points, out of): first-try answers on the checked steps."""
    pts, total = 0, 0
    for k in ("predict", "fill", "tweak"):
        if rec.get(k) is not None:
            total += 1
            pts += bool(rec[k])
    if rec.get("order"):
        total += 1
        pts += rec["order"]["tries"] == 1
    if rec.get("bug"):
        total += 1
        pts += rec["bug"]["found"]
    return pts, total


def run_lesson(stem: str, redo: bool = False, seed=None) -> dict:
    L = lesson(stem)
    if not L:
        print(f"    No lesson for {stem} yet.")
        return None
    rng = random.Random(seed if seed is not None else os.environ.get("CAVA_SEED"))
    progress = state.load(state.PROGRESS)
    entry = progress.setdefault(stem, {})
    entry.setdefault("opened_at", state.now())
    entry["mode"] = "read"
    started = state.now()
    rec = {}
    title = ast.get_docstring(ast.parse((PRACTICE / "drills" / f"{stem}.py").read_text(encoding="utf-8"))) or stem
    print(f"\n  {stem}" + ("  (redo)" if redo else ""))
    say(title.split("TASK")[0].strip(), "  ")
    if not redo:
        do_concept(stem, L, rec, rng)
    for fn in (do_predict, do_read, do_order, do_fill, do_bug, do_tweak):
        if redo and fn is do_read:
            continue
        fn(stem, L, rec, rng)
        state.touch()
    if not redo:
        do_explain(stem, L, rec, rng, entry)
    pts, total = score(rec)
    minutes = round((state.now() - started) / 60, 1)
    rec.update({"on": state.today().isoformat(), "minutes": minutes, "score": [pts, total]})
    entry.setdefault("read_log", []).append(rec)
    print(f"\n  {stem}: {pts}/{total} right first time · {minutes} min")
    return progress, entry, rec


# ── proving a lesson is right (selftest.py runs this on every lesson) ─

def validate(stem: str) -> list:
    """Problems with a lesson: anything that would teach something false."""
    L, src, out = lesson(stem), solution_source(stem), []
    if not L:
        return [f"{stem}: no lesson"]
    ok, msg = run_tests(stem, src)
    if not ok:
        return [f"{stem}: the reference solution itself fails: {msg}"]
    lines = {l.strip() for l in src.split("\n")}
    for key in (L.get("lines") or {}):
        if key.strip() not in lines:
            out.append(f"lines: '{key}' is not a line of the solution")
    if L.get("probe"):
        try:
            probe_value(stem, L["probe"])
        except Exception as e:
            out.append(f"probe: {type(e).__name__}: {e}")
    if L.get("order"):
        o = L["order"]
        body = o.get("lines") or [l.strip() for l in function_body(src, o["func"])[3] if l.strip()]
        ok, msg = run_tests(stem, with_body(src, o["func"], body))
        if not ok:
            out.append(f"order: the lines in the given order fail: {msg}")
        if len(body) < 3:
            out.append("order: fewer than 3 lines is not much of a puzzle")
    for f in L.get("fill") or []:
        try:
            if not run_tests(stem, replace_line(src, f["line"], f["line"]))[0]:
                out.append(f"fill: the right line fails: {f['line']}")
            for w in f["wrong"]:
                if run_tests(stem, replace_line(src, f["line"], w))[0]:
                    out.append(f"fill: a wrong option passes the tests: {w}")
        except ValueError as e:
            out.append(f"fill: {e}")
    for b in L.get("bugs") or []:
        try:
            if run_tests(stem, replace_line(src, b["line"], b["bad"]))[0]:
                out.append(f"bug: the planted bug passes the tests: {b['bad']}")
        except ValueError as e:
            out.append(f"bug: {e}")
    t = L.get("tweak")
    if t:
        if not 0 <= t["answer"] < len(t["options"]):
            out.append("tweak: answer index out of range")
        if t.get("run"):
            mod = load_source(src, f"tweak_{stem}")
            ns = {"np": np, "cv2": cv2, "synth": synth, "m": mod, "show": lambda *a: None,
                  "say": lambda *a: None, "demo_image": demo_image}
            ns.update({k: getattr(mod, k) for k in dir(mod) if not k.startswith("_")})
            try:
                exec(t["run"], ns)
            except Exception as e:
                out.append(f"tweak run: {type(e).__name__}: {e}")
    return [f"{stem}: {p}" for p in out]
