"""Study sessions: the clock, the weekly total, and what happens when one ends.

Only active time counts: the clock pauses after `idle_minutes` with no
activity (a checker run, a review answer, a coach choice, or a saved drill
file). A session shorter than `min_session_minutes` is kept in the log but
counts as zero. When a counted session ends, its minutes go to dra-solo and
me/ is pushed to its private repo.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from . import state

KINDS = ["drill", "review", "mock", "tema", "capstone"]
TIMER = Path(__file__).resolve().parent / "timer.py"
ACTIVITY = "activity.json"
INGEST_URL = "https://dra-solo.com/api/ingest"


# ── pure clock rules (tested in selftest.py) ─────────────────────────

def tick(s: dict, now: float, last_activity: float, idle_seconds: float) -> dict:
    """Advance the active clock to `now`; time past the idle limit doesn't count."""
    last = s.get("last_tick", now)
    if now > last:
        counted_until = min(now, last_activity + idle_seconds)
        if counted_until > last:
            s["active_seconds"] = s.get("active_seconds", 0) + (counted_until - last)
    s["last_tick"] = now
    s["idle"] = now - last_activity >= idle_seconds
    return s


def week_bounds(day: dt.date):
    monday = day - dt.timedelta(days=day.weekday())
    return monday, monday + dt.timedelta(days=7)


def week_summary(sessions: list, day: dt.date, live_minutes: float = 0, live_kind: str = "") -> dict:
    lo, hi = week_bounds(day)
    total, by_kind, count = 0.0, {}, 0
    for r in sessions:
        d = dt.date.fromisoformat(r["date"])
        if lo <= d < hi and r.get("counted"):
            total += r["minutes"]
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + r["minutes"]
            count += 1
    if live_minutes:
        total += live_minutes
        by_kind[live_kind] = by_kind.get(live_kind, 0) + live_minutes
    return {"minutes": total, "by_kind": by_kind, "sessions": count}


def fmt(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60}h{m % 60:02d}" if m >= 60 else f"{m}m"


# ── live session ─────────────────────────────────────────────────────

def current() -> dict:
    s = state.load(state.SESSION)
    return s if s.get("active") else {}


def last_activity() -> float:
    return state.load(ACTIVITY).get("at", 0)


def mark_activity() -> None:
    state.save(ACTIVITY, {"at": state.now()})


def live_minutes() -> float:
    s = current()
    if not s:
        return 0
    cfg = state.config()
    tick(s, state.now(), last_activity(), cfg["idle_minutes"] * 60)
    return s.get("active_seconds", 0) / 60


def start(kind: str = "drill", drill: str = "", window: bool = True) -> dict:
    s = current()
    if s:
        return s
    now = state.now()
    s = {"active": True, "kind": kind, "drill": drill, "started": now, "last_tick": now,
         "active_seconds": 0}
    state.save(state.SESSION, s)
    mark_activity()
    if window and os.environ.get("CAVA_NO_WINDOW") != "1":
        open_window()
    return s


def open_window() -> None:
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe") if exe.name.lower() == "python.exe" else exe
    flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0   # DETACHED_PROCESS | NEW_PROCESS_GROUP
    subprocess.Popen([str(pyw if pyw.exists() else exe), str(TIMER)], creationflags=flags,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     cwd=str(TIMER.parent.parent))


def set_drill(drill: str) -> None:
    s = current()
    if s and s.get("drill") != drill:
        s["drill"] = drill
        state.save(state.SESSION, s)


def finish() -> dict:
    """End the live session: log it, and if it counts, sync to dra-solo and back up."""
    s = current()
    if not s:
        return {}
    cfg = state.config()
    tick(s, state.now(), last_activity(), cfg["idle_minutes"] * 60)
    minutes = round(s.get("active_seconds", 0) / 60, 1)
    rec = {"date": dt.date.fromtimestamp(s["started"]).isoformat(), "start": s["started"],
           "end": state.now(), "kind": s["kind"], "drill": s.get("drill", ""),
           "minutes": minutes, "counted": minutes >= cfg["min_session_minutes"]}
    sessions = state.load(state.SESSIONS, default=[])
    sessions.append(rec)
    state.save(state.SESSIONS, sessions)
    state.save(state.SESSION, {"active": False, "last": rec})
    status = state.load(state.BACKUP)
    if rec["counted"]:
        status.setdefault("dra_pending", []).append({"minutes": minutes, "kind": rec["kind"], "date": rec["date"]})
    status["dra_pending"] = sync_dra(status.get("dra_pending", []))
    status["backup_pending"] = not push_backup(rec)
    state.save(state.BACKUP, status)
    return rec


# ── dra-solo and the private backup ──────────────────────────────────

def sync_dra(pending: list) -> list:
    """Send counted sessions to the `cava` habit on dra-solo. Returns what is still pending."""
    token = os.environ.get("DRASOLO_INGEST_TOKEN")
    if not token or not pending:
        return pending
    left = []
    for p in pending:
        body = json.dumps({"token": token, "kind": "habit", "source": "cava",
                           "payload": {"habit": "cava", "value": p["minutes"], "note": p["kind"]}}).encode()
        req = urllib.request.Request(INGEST_URL, data=body, method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": "cava-coach"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status != 200:
                    left.append(p)
        except OSError:
            left.append(p)
    return left


def push_backup(rec: dict) -> bool:
    """Commit and push me/ to its private repo. True when it is up to date."""
    if not (state.ME / ".git").exists():
        return True    # no backup repo configured: nothing to be pending about
    def git(*a):
        return subprocess.run(["git", "-C", str(state.ME), *a], capture_output=True, text=True, timeout=60)
    try:
        git("add", "-A")
        if git("diff", "--cached", "--quiet").returncode != 0:
            git("commit", "-q", "-m", f"session {rec.get('date', '')} {rec.get('kind', '')} {rec.get('minutes', '')} min")
        return git("push", "-q").returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False
