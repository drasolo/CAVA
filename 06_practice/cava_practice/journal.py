"""me/journal.md, regenerated from progress.json after every run.

It is the revision sheet: per drill, what you predicted, the traps you hit, how
you explained it, and how the redos went. Edit it freely; it is rebuilt from
the data, so keep anything you want to last in the data (or a separate file).
"""
import datetime as dt

from . import state


def _when(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""


def render(progress: dict) -> str:
    out = ["# CAVA journal", "",
           "_Rebuilt from progress.json after every checker run. Newest pass first._", ""]
    passed = sorted(((e.get("passed_at", 0), s) for s, e in progress.items() if e.get("passed")), reverse=True)
    going = sorted(s for s, e in progress.items() if not e.get("passed") and e.get("run_log"))
    if going:
        out += ["## In progress", ""]
        for s in going:
            e = progress[s]
            out.append(f"- **{s}**: {len(e['run_log'])} run(s), last: `{e['run_log'][-1]['msg'][:80]}`")
        out.append("")
    for _, s in passed:
        e = progress[s]
        out += [f"## {s}", "",
                f"passed {e['passed']} · {e.get('minutes', '?')} min · {e.get('runs_to_pass', '?')} run(s) · "
                f"{e.get('phase', '?')}" + (" · assisted" if e.get("assisted") else "")
                + (f" · {len(e['hints'])} hint(s)" if e.get("hints") else ""), ""]
        for k, r in enumerate(e.get("read_log", [])):
            pts, total = r.get("score", [0, 0])
            missed = [s for s in ("predict", "fill", "tweak") if r.get(s) is False]
            if r.get("order") and r["order"]["tries"] > 1:
                missed.append("order")
            if r.get("bug") and not r["bug"]["found"]:
                missed.append("bug")
            out.append(f"- **{'Lesson' if k == 0 else 'Redo'} {r.get('on', '')}:** {pts}/{total} first try"
                       + (f" · missed: {', '.join(missed)}" if missed else ""))
            if r.get("bug", {}) and r["bug"].get("why"):
                out.append(f"  - why the bug was wrong, in my words: {r['bug']['why']}")
        p = e.get("prediction")
        if p:
            right = {True: "right", False: "wrong"}.get(e.get("prediction_right"), "?")
            out.append(f"- **Predicted** ({right}): {p.get('plan')} · trap: {p.get('trap')} · value: {p.get('value')}")
        whys = [r["why"] for r in e.get("run_log", []) if r.get("why")]
        for w in whys:
            out.append(f"- **Trap hit:** {w}")
        if e.get("explanation"):
            out.append(f"- **In my words:** {e['explanation']}")
        if e.get("card"):
            out.append(f"- **Card:** {e['card']['q']} → {e['card']['a']}")
        redos = e.get("redo_log", [])
        if redos:
            out.append("- **Redos:** " + ", ".join(
                f"{r['on']} {r['minutes']} min {'✓' if r['ok'] else '✗'}" for r in redos))
        pending = [d for d in e.get("redos", []) if d not in {r["due"] for r in redos}]
        if pending:
            out.append(f"- **Next redo:** {pending[0]}")
        out.append("")
    return "\n".join(out)


def write() -> None:
    state.path("journal.md").write_text(render(state.load(state.PROGRESS)), encoding="utf-8")
