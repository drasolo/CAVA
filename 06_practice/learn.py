"""Onboarding: one drill, start to finish, with the whole learning flow.

    python learn.py          d00 (or the first drill you haven't passed)
    python learn.py d01      a specific drill

Every later drill uses the same steps, lighter. Read the README's "The flow"
section for the why behind each one.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import check  # noqa: E402
from cava_practice import concepts, flow, session, state  # noqa: E402


def say(text: str) -> None:
    print()
    for line in text.strip().splitlines():
        print(f"  {line}")


def pause(prompt: str = "press Enter to go on") -> str:
    return flow.ask(f"\n  ({prompt}) ", False)


def main():
    progress = state.load(state.PROGRESS)
    stems = flow.stems()
    if len(sys.argv) > 1:
        stem = next((s for s in stems if s.startswith(sys.argv[1])), None)
        if not stem:
            sys.exit(f"No drill matches {sys.argv[1]}.")
    else:
        stem = next((s for s in stems if not progress.get(s, {}).get("passed")), stems[0])
    ph = flow.phase(stem, progress)

    say(f"""
Welcome. This walks you through {stem} with the full flow, once.
Every step is there because it makes things stick, not to slow you down:

  1 read the task, not the hints      6 compare with the solution, after
  2 predict before coding             7 explain it in two sentences
  3 write it cold                     8 turn the trap into a card
  4 predict each run, explain fails   9 redo it from memory later
  5 hints only on a timer

A session starts now; the small window shows active time. It counts once it
reaches 20 minutes. Your week's total is under it.""")
    session.start("drill", stem)
    pause()

    say("STEP 1 · Read the task. The hints are hidden on purpose; you'll unlock them only if you need them.")
    entry = progress.setdefault(stem, {})
    entry.setdefault("opened_at", state.now())
    entry.setdefault("phase", ph)
    state.save(state.PROGRESS, progress)
    print()
    for line in check.statement(stem).splitlines():
        print(f"    {line}")
    pause("read it, then Enter")

    say("STEP 2 · Predict. A wrong guess now makes the right answer stick far better than reading it.")
    flow.predict(entry, ph)
    state.save(state.PROGRESS, progress)

    say(f"""
STEP 3 · Write it cold. Open drills/{stem}.py in your editor.
Allowed: the NumPy / OpenCV docs, the way the exam allows them.
Not allowed: the solution, old code, AI. Being stuck for a while is the useful part.
When you think it's right, come back here.""")

    say("STEP 4 · Each run: say what you expect first. If it fails, write why before the next run.")
    while True:
        key = pause("Enter to run the checker · h = a hint · c = I don't get the concept · q = stop for now").lower()[:1]
        if key == "q":
            say(f"Stopped. Continue any time with `python learn.py {stem[:3]}` or `python check.py {stem[:3]}`.")
            return
        if key == "h":
            say("STEP 5 · Hints come one at a time, and in the strict phase only after 15 minutes.")
            check.hint(stem)
            continue
        if key == "c":
            concepts.descend(stem)
            continue
        if check.attempt(stem):
            break
        say("Not yet. That message is the most useful thing on the screen: what exactly differs?")

    progress = state.load(state.PROGRESS)
    entry = progress[stem]
    say(f"""
STEP 6 · Now (and only now) compare: open solutions/{stem}.py.
Look for one idea that's better than yours. Then close it.""")
    better = flow.ask("  One better idea you saw (or 'mine was fine'): ", False)
    if better:
        entry["compare_note"] = better
    say(f"""
STEPS 7 and 8 · done: your two sentences and your card are saved. The card joins
the review queue (`python theory.py`).

STEP 9 · The redo. On {entry['redos'][0]} the coach will ask you to write {stem} again from
memory, with the clock running: your code gets archived and the blank stub comes back.
The goal is one run and a third of today's time.""")
    cfg = state.load(state.CONFIG)
    cfg["onboarded"] = state.today().isoformat()
    state.save(state.CONFIG, {**state.config(), **cfg})
    state.save(state.PROGRESS, progress)

    say("""
That's the flow. From here, `cava` (or `python coach.py`) is your start point:
tell it how you feel and it picks the next step. The session keeps running
until you stop it there or with the window's stop button.""")


if __name__ == "__main__":
    main()
