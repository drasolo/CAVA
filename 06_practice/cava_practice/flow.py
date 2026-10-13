"""The learning flow around each drill: phases, hints, notes, redos.

Pure rules live at the top (tested by selftest.py); the prompts that talk to
you are at the bottom. Nothing here runs a drill: check.py does that and asks
this module what is allowed and what to record.

The first STRICT_COUNT drills are strict: you write a prediction before the
checker will run, a one-line expectation before every run, a note on why the
last run failed before the next one, and a card and two sentences before the
pass is recorded. Hints unlock HINT_LOCK_MIN minutes after you first open the
drill, one step at a time. After that the rules relax (see LIGHT below), and
they relax early once three strict drills in a row went smoothly.
"""
import datetime as dt
import re
from pathlib import Path

from . import state

PRACTICE = Path(__file__).resolve().parent.parent
DRILLS = PRACTICE / "drills"
HINTS = PRACTICE / "hints"

STRICT_COUNT = 10
HINT_LOCK_MIN = 15
STRICT_REDOS = [1, 7]      # days after the pass; plus one a week before the exam
LIGHT_REDOS = [3]
ASSISTED_FIRST_REDO = 1    # a pass with tutor help comes back after a day


def stems():
    return sorted(p.stem for p in DRILLS.glob("d[0-9][0-9]_*.py"))


def strict_stems():
    return stems()[:STRICT_COUNT]


# ── phase ─────────────────────────────────────────────────────────────

def smooth(entry: dict) -> bool:
    """A strict drill that went well: passed in two runs or fewer, prediction right."""
    return bool(entry.get("passed")) and entry.get("runs_to_pass", 99) <= 2 \
        and entry.get("prediction_right") is True


def early_exit(progress: dict) -> bool:
    """Three strict drills in a row, in the order you passed them, went smoothly."""
    passed = sorted((e["passed_at"], s) for s, e in progress.items()
                    if s in strict_stems() and e.get("passed_at"))
    run = 0
    for _, s in passed:
        run = run + 1 if smooth(progress[s]) else 0
        if run >= 3:
            return True
    return False


def phase(stem: str, progress: dict) -> str:
    if stem in strict_stems() and not early_exit(progress):
        return "strict"
    return "light"


# ── hints ─────────────────────────────────────────────────────────────

def hint_steps(stem: str) -> list:
    """The hint ladder for a drill: one step per numbered idea or sentence."""
    f = HINTS / f"{stem}.md"
    if not f.exists():
        return []
    text = f.read_text(encoding="utf-8").strip()
    # Rejoin wrapped lines; a numbered idea or a matrix row like "[x, y, 1]" starts a new one.
    joined = []
    for line in text.splitlines():
        if joined and line.strip() and not re.match(r"\s*(\d+\.\s|\[)", line):
            joined[-1] = joined[-1].rstrip() + " " + line.strip()
        else:
            joined.append(line)
    text = "\n".join(joined)
    items = [i.strip() for i in re.split(r"\n\s*(?=\d+\.\s)", text) if i.strip()]
    if len(items) > 1:
        if not re.match(r"\d+\.", items[0]):   # a preamble like "(pick one)" leads the first idea
            items = [items[0] + " " + items[1]] + items[2:]
        return items
    parts = re.split(r"(?<=[A-Za-z0-9\)\]])\.\s+(?=[A-Z])", text)
    return [p.strip() + ("" if p.strip().endswith((".", ":")) else ".") for p in parts if p.strip()]


def hint_unlocked_at(entry: dict, ph: str) -> float:
    """Epoch seconds when the next hint may be shown (0 = now)."""
    if ph != "strict":
        return 0
    return entry.get("opened_at", state.now()) + HINT_LOCK_MIN * 60


# ── redos ─────────────────────────────────────────────────────────────

def schedule_redos(passed: dt.date, ph: str, assisted: bool, exam: dt.date) -> list:
    days = list(STRICT_REDOS if ph == "strict" else LIGHT_REDOS)
    if assisted:
        days[0] = ASSISTED_FIRST_REDO
    dates = [passed + dt.timedelta(days=d) for d in days]
    pre_exam = exam - dt.timedelta(days=7)
    if pre_exam > dates[-1]:
        dates.append(pre_exam)
    return [d.isoformat() for d in dates]


def due_redo(entry: dict, today: dt.date):
    """The earliest redo date that is due and not done yet, or None.

    A redo that went badly still uses up its date: check.py schedules the retry
    for the next day instead."""
    done = {r["due"] for r in entry.get("redo_log", [])}
    for d in entry.get("redos", []):
        if d not in done and dt.date.fromisoformat(d) <= today:
            return d
    return None


def redo_went_well(runs: int, minutes: float, first_minutes: float) -> bool:
    """A from-memory redo counts when it passed first time, in a third of the original time."""
    return runs <= 1 and (not first_minutes or minutes <= max(5, first_minutes / 3))


# ── the prompts ───────────────────────────────────────────────────────

def ask(prompt: str, required: bool) -> str:
    while True:
        try:
            ans = input(prompt).strip()
        except EOFError:
            ans = ""
        if ans or not required:
            return ans
        print("    (this one is required in the strict phase)")


def predict(entry: dict, ph: str) -> None:
    """Before the first run: plan, the trap you expect, and one concrete value."""
    if entry.get("prediction"):
        return
    strict = ph == "strict"
    print("\n    Before any code: predict. Wrong guesses are fine, they make the answer stick.")
    plan = ask("    Your plan, in a few words: ", strict)
    trap = ask("    One trap you expect: ", strict)
    value = ask("    One concrete value you expect (e.g. pure red -> 76): ", strict)
    if plan or trap or value:
        entry["prediction"] = {"plan": plan, "trap": trap, "value": value, "at": state.now()}


def expect(entry: dict, ph: str) -> str:
    """Before each run: what you think the checker will say."""
    if ph != "strict":
        return ""
    return ask("    Before you run: what do you expect it to say? ", True)


def needs_failure_note(entry: dict, ph: str) -> bool:
    runs = entry.get("run_log", [])
    return ph == "strict" and bool(runs) and not runs[-1]["ok"] and not runs[-1].get("why")


def failure_note(entry: dict) -> None:
    last = entry["run_log"][-1]
    print(f"\n    Last run: {last['msg']}")
    last["why"] = ask("    Why did it fail, in one line? (this is the part you learn) ", True)


def on_pass(entry: dict, ph: str) -> bool:
    """After the first PASS: check the prediction, two sentences, a card. True when recorded."""
    print("\n    PASS. Before it counts:")
    if entry.get("prediction"):
        p = entry["prediction"]
        print(f"    You predicted: plan '{p['plan']}', trap '{p['trap']}', value '{p['value']}'.")
        entry["prediction_right"] = ask("    Was your prediction right? [y/n] ", True).lower().startswith("y")
    print("    Explain it in two sentences: what it does, and why it matters later.")
    one = ask("    1. ", True)
    two = ask("    2. ", True)
    entry["explanation"] = f"{one} {two}".strip()
    strict = ph == "strict"
    print("    Turn the trap into a card for later review." + ("" if strict else " (Enter to skip)"))
    q = ask("    Card question: ", strict)
    if q:
        a = ask("    Card answer: ", True)
        entry["card"] = {"q": q, "a": a}
    return True
