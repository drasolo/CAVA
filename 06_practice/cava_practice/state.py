"""Personal state in me/: small JSON files, written atomically.

Every file here is yours, not the repo's. The coach commits me/ to its own
private repo at the end of a session.
"""
import datetime as dt
import json
import os
from pathlib import Path

from .paths import ME

PROGRESS = "progress.json"        # per drill: runs, predictions, notes, pass, redos
THEORY = "theory_progress.json"   # per review item: Leitner box and due date
CARDS = "cards.json"              # your own review questions, one per drill
SESSIONS = "sessions.json"        # finished study sessions
SESSION = "session.json"          # the live session, shared with the timer window
CONFIG = "config.json"            # weekly target, milestones, exam date
FLAGS = "flags.json"              # concept explanations that didn't help
BACKUP = "backup.json"            # last backup / dra-solo sync status

DEFAULT_CONFIG = {
    "weekly_minutes": 240,
    "min_session_minutes": 20,
    "idle_minutes": 10,
    # Before each Tema comes out (last year: 12 Nov and 17 Dec); edit when the real dates are known.
    "milestones": [
        {"by": "2026-11-10", "levels": [0, 1, 2, 3, 4], "why": "Tema 1"},
        {"by": "2026-12-15", "levels": [5, 6, 7], "why": "Tema 2"},
    ],
    "exam": "2027-02-01",
    "tutor_in_tema": "concepts",   # "concepts" or "off"
    # "read": learn each drill from its working code (lessons/), "write": write the code yourself.
    "mode": "read",
}


def path(name: str) -> Path:
    ME.mkdir(parents=True, exist_ok=True)
    return ME / name


def load(name: str, default=None):
    try:
        return json.loads(path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


def save(name: str, data) -> None:
    p = path(name)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(load(CONFIG))
    if not path(CONFIG).exists():
        save(CONFIG, cfg)
    return cfg


def today() -> dt.date:
    # CAVA_TODAY lets tests (and you) check what is due on another day.
    fake = os.environ.get("CAVA_TODAY")
    return dt.date.fromisoformat(fake) if fake else dt.date.today()


def now() -> float:
    return dt.datetime.now().timestamp()


def touch() -> None:
    """Mark activity, so the session clock keeps running. A file of its own, so the
    timer window (which owns session.json) and the checker never overwrite each other."""
    save("activity.json", {"at": now()})
