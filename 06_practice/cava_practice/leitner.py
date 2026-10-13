"""A five-box Leitner schedule for review items (theory questions and cards).

stats[key] = {"box": 0-4, "due": "YYYY-MM-DD", "seen": n, "right": n}
An item with no "box" has never been answered: it is new, not due.
"""
import datetime as dt

INTERVALS = [1, 3, 7, 14, 30]   # days until the next review, by box


def migrate(stats: dict) -> dict:
    """Old theory_progress.json entries ({seen, right}) get a box from their hit rate, due today."""
    today = dt.date.today().isoformat()
    for s in stats.values():
        if "box" not in s and s.get("seen"):
            rate = s.get("right", 0) / s["seen"]
            s["box"] = 0 if rate < 0.5 else (1 if rate < 0.8 else 2)
            s["due"] = today
    return stats


def due_keys(stats: dict, keys: list, today: dt.date) -> list:
    due = [(stats[k]["due"], k) for k in keys
           if k in stats and "box" in stats[k] and dt.date.fromisoformat(stats[k]["due"]) <= today]
    return [k for _, k in sorted(due)]


def pick(stats: dict, keys: list, today: dt.date, n: int, rng) -> list:
    """Due items first, oldest due first; then new items, shuffled."""
    chosen = due_keys(stats, keys, today)[:n]
    new = [k for k in keys if "box" not in stats.get(k, {})]
    rng.shuffle(new)
    return chosen + new[:max(0, n - len(chosen))]


def answer(stats: dict, key: str, ok: bool, today: dt.date) -> None:
    s = stats.setdefault(key, {"seen": 0, "right": 0})
    s["seen"] = s.get("seen", 0) + 1
    s["right"] = s.get("right", 0) + int(ok)
    if ok:
        s["box"] = min(len(INTERVALS) - 1, s["box"] + 1) if "box" in s else 1
    else:
        s["box"] = 0
    s["due"] = (today + dt.timedelta(days=INTERVALS[s["box"]])).isoformat()
