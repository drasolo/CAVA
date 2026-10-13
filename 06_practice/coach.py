"""The coach: tell it how you feel, it picks the next action and starts it.

    python coach.py           the menu (or type `cava` anywhere, see README)
    python coach.py --list    every feeling and what it does
    python coach.py stop      end the running session

It also keeps the session clock (a small window), the week's total against
your target, where you stand against the next Tema milestone, and what review
items and redos are due.
"""
import datetime as dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cava_practice import concepts, flow, session, state, tutor  # noqa: E402

PY = sys.executable

# (key, how you feel, what the coach does) -- the --list table and the menu.
FEELINGS = [
    ("1", "I can't get started", "5-minute starter: only the 'predict' step of your current drill"),
    ("2", "I've been stuck for 15+ minutes", "the next hint step (strict drills unlock hints after 15 min)"),
    ("3", "My output differs and I don't see why", "re-run the checker and read the failing case (visual diff: phase 2)"),
    ("4", "I don't get the concept", "walk the concept map down to something you have, then back up"),
    ("5", "I'm overwhelmed", "shrink it: one function, its part of the task, a 10-minute goal"),
    ("6", "I'm bored", "a few review items now (webcam playground: phase 2)"),
    ("7", "I'm tired", "10 review items, no new code"),
    ("8", "I only have 15 minutes", "your due review items (cards and theory)"),
    ("9", "I'm anxious about the exam", "a timed from-memory redo of a drill you passed"),
    ("0", "I'm on fire", "the next drill (or the redo that is due)"),
    ("f", "I just finished a drill", "teach-back: Claude asks you one probing question"),
]
OTHER = [
    ("s", "start a session (drill / review / mock / tema / capstone)"),
    ("x", "stop the session"),
    ("l", "list the drills"),
    ("j", "open the journal"),
    ("o", "onboarding again (learn.py)"),
    ("q", "quit (the session keeps running until you stop it)"),
]


def run(*args):
    subprocess.run([PY, *args], cwd=ROOT)
    state.touch()


def current_drill(progress: dict) -> str:
    for s in flow.stems():
        if progress.get(s, {}).get("redo_active"):
            return s
    for s in flow.stems():
        if not progress.get(s, {}).get("passed"):
            return s
    return flow.stems()[-1]


def due_review() -> int:
    out = subprocess.run([PY, "theory.py", "--due"], cwd=ROOT, capture_output=True, text=True)
    try:
        return int(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return 0


def milestone_line(progress: dict, cfg: dict) -> str:
    today = state.today()
    for m in cfg["milestones"]:
        by = dt.date.fromisoformat(m["by"])
        if by < today:
            continue
        stems = [s for s in flow.stems() if int(s[1]) in m["levels"]]
        done = sum(1 for s in stems if progress.get(s, {}).get("passed"))
        start = dt.date.fromisoformat(cfg.get("plan_start", "2026-10-13"))
        span = max(1, (by - start).days)
        expected = round(len(stems) * min(1, max(0, (today - start).days) / span))
        gap = expected - done
        pace = "on track" if gap <= 0 else f"{gap} behind"
        return f"{m['why']}: levels {m['levels'][0]}-{m['levels'][-1]} by {m['by']} · {done}/{len(stems)} · {pace}"
    return "all milestones passed: time for the mocks"


def target_hint(sessions: list, target: int) -> str:
    """After two full weeks, suggest a target that matches what you actually do."""
    counted = [s for s in sessions if s.get("counted")]
    if not counted:
        return ""
    first = min(dt.date.fromisoformat(s["date"]) for s in counted)
    monday = state.today() - dt.timedelta(days=state.today().weekday())
    weeks = (monday - (first - dt.timedelta(days=first.weekday()))).days // 7
    if weeks < 2:
        return ""
    lo = monday - dt.timedelta(weeks=2)
    avg = sum(s["minutes"] for s in counted if lo <= dt.date.fromisoformat(s["date"]) < monday) / 2
    if abs(avg - target) <= 0.25 * target:
        return ""
    suggest = int(round(max(60, avg) / 30) * 30)
    return (f"tip: the last two weeks averaged {session.fmt(avg)}; a target of {session.fmt(suggest)} "
            f"would fit (weekly_minutes in me/config.json)")


def header() -> str:
    cfg = state.config()
    progress = state.load(state.PROGRESS)
    live = session.current()
    live_min = session.live_minutes() if live else 0
    counts = live_min >= cfg["min_session_minutes"]
    wk = session.week_summary(state.load(state.SESSIONS, default=[]), state.today(),
                              live_min if counts else 0, live.get("kind", ""))
    target = cfg["weekly_minutes"]
    filled = min(10, int(10 * wk["minutes"] / target)) if target else 0
    split = " · ".join(f"{k} {session.fmt(v)}" for k, v in sorted(wk["by_kind"].items()))
    lines = [f"\n  cava · week {session.fmt(wk['minutes'])} / {session.fmt(target)}  "
             f"{'▓' * filled}{'░' * (10 - filled)}" + (f"  ({split})" if split else "")]
    cur = current_drill(progress)
    due_redos = [s for s, e in progress.items() if e.get("passed") and flow.due_redo(e, state.today())]
    tag = f" ({flow.phase(cur, progress)})" if not progress.get(cur, {}).get("passed") else ""
    lines.append(f"  now: {cur}{tag} · {due_review()} review due"
                 + (f" · redo due: {', '.join(due_redos)}" if due_redos else ""))
    lines.append(f"  {milestone_line(progress, cfg)}")
    if live:
        lines.append(f"  session: {live['kind']} running, {session.fmt(live_min)} active"
                     + ("" if counts else f" (counts from {cfg['min_session_minutes']} min)"))
    hint = target_hint(state.load(state.SESSIONS, default=[]), target)
    if hint:
        lines.append(f"  {hint}")
    status = state.load(state.BACKUP)
    if status.get("backup_pending") or status.get("dra_pending"):
        lines.append("  ! " + " · ".join(x for x in (
            "backup pending" if status.get("backup_pending") else "",
            f"{len(status['dra_pending'])} session(s) not on dra-solo yet" if status.get("dra_pending") else "")
            if x))
    return "\n".join(lines)


def ensure_session(kind: str, drill: str = "") -> None:
    if not session.current():
        session.start(kind, drill)
        print(f"    session started ({kind}). The clock window shows active time.")
    elif drill:
        session.set_drill(drill)


# ── the actions ──────────────────────────────────────────────────────

def starter(stem):
    ensure_session("drill", stem)
    progress = state.load(state.PROGRESS)
    entry = progress.setdefault(stem, {})
    entry.setdefault("opened_at", state.now())
    print(f"\n{concepts_statement(stem)}\n")
    flow.predict(entry, flow.phase(stem, progress))
    state.save(state.PROGRESS, progress)
    print(f"\n    That's the start. Now 5 minutes on {stem}.py, nothing more required.")
    print(f"    When you want to check: `python check.py {stem[:3]}`.")


def concepts_statement(stem):
    import ast
    return ast.get_docstring(ast.parse((ROOT / "drills" / f"{stem}.py").read_text(encoding="utf-8"))) or ""


def shrink(stem):
    """Overwhelmed: the first function still to write, and only its part of the task."""
    import ast
    src = (ROOT / "drills" / f"{stem}.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    todo = [f.name for f in tree.body if isinstance(f, ast.FunctionDef)
            and any(isinstance(n, ast.Raise) for n in ast.walk(f))]
    names = todo or [f.name for f in tree.body if isinstance(f, ast.FunctionDef)]
    if not names:
        print("    Nothing to shrink here.")
        return
    name = names[0]
    doc = ast.get_docstring(tree) or ""
    lines, keep = doc.splitlines(), []
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{name}("):
            indent = len(line) - len(line.lstrip())
            keep.append(line)
            for nxt in lines[i + 1:]:      # its description is indented deeper than the signature
                if nxt.strip() and len(nxt) - len(nxt.lstrip()) > indent:
                    keep.append(nxt)
                else:
                    break
            break
    ensure_session("drill", stem)
    print(f"\n    Just this, for 10 minutes. Ignore everything else in {stem}.py:\n")
    print("\n".join(keep) if keep else f"    {name}()")
    print("\n    Write only that function. Then run the checker: the rest will still say TODO, that's fine.")


def teach_back(progress):
    passed = sorted(((e.get("passed_at", 0), s) for s, e in progress.items() if e.get("passed")), reverse=True)
    if not passed:
        print("    Pass a drill first; then I'll quiz you on it.")
        return
    stem = passed[0][1]
    e = progress[stem]
    print(f"\n    {stem}. You explained it as: {e.get('explanation', '(no explanation recorded)')}")
    q = tutor.ask(f"A student just solved a drill: '{concepts_statement(stem).splitlines()[0]}'. "
                  f"Their own explanation: '{e.get('explanation', '')}'. Act as a confused classmate and ask ONE "
                  "probing question that tests whether they really understand why it works. Only the question.")
    print(f"\n    classmate: {q}")
    ans = flow.ask("    you: ", True)
    fb = tutor.ask(f"Question: {q}\nStudent's answer: {ans}\nIn 2-3 sentences: is it right, what is missing? No code.")
    print(f"\n    {fb}")


def timed_redo(progress):
    passed = sorted(((e.get("passed_at", 0), s) for s, e in progress.items() if e.get("passed")))
    if not passed:
        print("    No passed drill to redo yet. Here are 10 review items instead.")
        ensure_session("review")
        run("theory.py", "10")
        return
    stem = passed[0][1]
    ensure_session("drill", stem)
    print(f"    Timed redo of {stem}: your code gets archived, the stub comes back, the clock runs.")
    run("check.py", stem[:3], "--redo")


def act(key: str) -> None:
    progress = state.load(state.PROGRESS)
    cur = current_drill(progress)
    if key == "1":
        starter(cur)
    elif key == "2":
        ensure_session("drill", cur)
        run("check.py", cur[:3], "--hint")
    elif key == "3":
        ensure_session("drill", cur)
        run("check.py", cur[:3])
        print("    Read the failing case: it names the input, your value and the expected one.")
        print("    Print your intermediate arrays on that exact case. (Image diff window: phase 2.)")
    elif key == "4":
        ensure_session("drill", cur)
        concepts.descend(cur)
    elif key == "5":
        shrink(cur)
    elif key in ("6", "7"):
        ensure_session("review")
        run("theory.py", "5" if key == "6" else "10")
    elif key == "8":
        ensure_session("review")
        run("theory.py", str(max(5, min(15, due_review()))))
    elif key == "9":
        timed_redo(progress)
    elif key == "0":
        ensure_session("drill", cur)
        run("check.py", "next")
    elif key == "f":
        teach_back(progress)
    elif key == "s":
        kind = flow.ask(f"    kind [{'/'.join(session.KINDS)}] (drill): ", False) or "drill"
        if kind not in session.KINDS:
            print(f"    unknown kind: {kind}")
            return
        if kind == "tema":
            mode = state.config().get("tutor_in_tema", "concepts")
            print(f"    Tema work is graded: the tutor is '{mode}' here (concepts only, never your Tema code).")
            print("    Check the statement's rules on AI help before using it at all.")
        ensure_session(kind, cur if kind == "drill" else "")
    elif key == "x":
        stop()
    elif key == "l":
        run("check.py")
    elif key == "j":
        print(state.path("journal.md").read_text(encoding="utf-8") if state.path("journal.md").exists()
              else "    The journal starts with your first checker run.")
    elif key == "o":
        run("learn.py")


def stop() -> None:
    rec = session.finish()
    if not rec:
        print("    No session running.")
        return
    cfg = state.config()
    verdict = "counted" if rec["counted"] else f"under {cfg['min_session_minutes']} min, not counted"
    print(f"    Session ended: {session.fmt(rec['minutes'])} active, {verdict}.")
    status = state.load(state.BACKUP)
    if status.get("backup_pending"):
        print("    Backup pending (offline?). It retries at the next session end.")


def table() -> str:
    w = max(len(f) for _, f, _ in FEELINGS)
    return "\n".join(f"  {k}  {f:<{w}}  -> {a}" for k, f, a in FEELINGS)


def main():
    if "--list" in sys.argv:
        print(table())
        return
    if "stop" in sys.argv:
        stop()
        return
    progress = state.load(state.PROGRESS)
    if not progress.get("d00_gray", {}).get("passed") and not state.load(state.CONFIG).get("onboarded"):
        print("\n  First time here: the onboarding walks you through one drill with the full flow.")
        if flow.ask("  Start it now? [Y/n] ", False).lower() != "n":
            run("learn.py")
            return
    while True:
        print(header())
        print("\n  How do you feel?")
        print(table())
        print()
        for k, label in OTHER:
            print(f"  {k}  {label}")
        key = flow.ask("\n  > ", False).lower()[:1]
        if key in ("q", ""):
            return
        state.touch()
        act(key)


if __name__ == "__main__":
    main()
