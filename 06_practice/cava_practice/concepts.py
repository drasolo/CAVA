"""Walk the concept map: down until a concept you already have, then back up.

For each concept you get its check question first. Had it: move on. Didn't:
go down to what it needs, then read its explanation and try the check again.
"explain differently" asks Claude for another angle; "didn't help" is logged
in me/flags.json so the explanation gets rewritten.
"""
from pathlib import Path

import yaml

from . import flow, state

GRAPH = Path(__file__).resolve().parent.parent / "concepts" / "graph.yaml"


def graph() -> dict:
    return yaml.safe_load(GRAPH.read_text(encoding="utf-8"))


def missing(g: dict) -> list:
    """Prerequisite ids that don't exist (the graph must stay closed)."""
    ids = set(g["concepts"])
    bad = [(c, n) for c, v in g["concepts"].items() for n in v.get("needs", []) if n not in ids]
    bad += [(d, n) for d, ns in g["drills"].items() for n in ns if n not in ids]
    return bad


def cycle(g: dict):
    """A concept that (indirectly) needs itself, or None."""
    seen, stack = set(), set()

    def visit(c):
        if c in stack:
            return c
        if c in seen:
            return None
        seen.add(c)
        stack.add(c)
        for n in g["concepts"][c].get("needs", []):
            hit = visit(n)
            if hit:
                return hit
        stack.discard(c)
        return None

    for c in g["concepts"]:
        hit = visit(c)
        if hit:
            return hit
    return None


def explain_differently(c: dict) -> None:
    from . import tutor
    tutor.explain(c["name"], c["explain"])


def flag(cid: str, stem: str) -> None:
    flags = state.load(state.FLAGS, default=[])
    flags.append({"concept": cid, "drill": stem, "at": state.today().isoformat()})
    state.save(state.FLAGS, flags)


def walk(cid: str, g: dict, stem: str, done: set, depth: int = 0) -> None:
    if cid in done:
        return
    c = g["concepts"][cid]
    pad = "    " + "  " * depth
    print(f"\n{pad}[{c['name']}]  {c['check']['q']}")
    flow.ask(f"{pad}Answer in your head or on paper, then press Enter ", False)
    print(f"{pad}-> {c['check']['a']}")
    if flow.ask(f"{pad}Did you have it? [y/n] ", True).lower().startswith("y"):
        done.add(cid)
        return
    for n in c.get("needs", []):
        walk(n, g, stem, done, depth + 1)
    while True:
        print(f"\n{pad}{c['name'].upper()}")
        for line in c["explain"].strip().split(". "):
            print(f"{pad}  {line.strip().rstrip('.')}.")
        choice = flow.ask(f"{pad}[g]ot it · [d]ifferent explanation · didn't [h]elp: ", False).lower()[:1]
        if choice == "d":
            explain_differently(c)
        elif choice == "h":
            flag(cid, stem)
            explain_differently(c)
        else:
            break
    state.touch()
    done.add(cid)


def descend(stem: str) -> None:
    g = graph()
    roots = g["drills"].get(stem)
    if not roots:
        print(f"    No concept map for {stem} yet. Use the tutor, or the course slides it names.")
        return
    print(f"\n    {stem}: let's find the first idea that isn't clear yet.")
    done = set()
    for r in roots:
        walk(r, g, stem, done)
    print(f"\n    Back to the drill: `python check.py {stem[:3]}`.")
