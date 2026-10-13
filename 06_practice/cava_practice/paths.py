"""Where the real datasets live.

Heavy data is not in the repo. It sits in the backup mirror on D:. Override the
location with the CAVA_BACKUP environment variable if you move it.
"""
import os
from pathlib import Path

PRACTICE = Path(__file__).resolve().parent.parent
REPO = PRACTICE.parent
BACKUP = Path(os.environ.get(
    "CAVA_BACKUP", r"D:\_QUARANTINE\2026-08-23\CAVA-ALL_BACKUP_2026-06-03"))
PAST = BACKUP / "CAVA" / "CAVA_2025_2026_NEW" / "Past_exams"

# Raw exam data, as handed out in the exam room.
EXAMS = {
    "2025-02-06": PAST / "343_GABROVEANU_RAZVAN_EXAMEN_CAVA" / "343_GABROVEANU_RAZVAN_EXAMEN_CAVA" / "data",
    "2025-02-20": PAST / "Subiect2" / "data",
    "2025-06-06": PAST / "Subiect3" / "Subiect3" / "data",
    "2026-02-06": BACKUP / "Lucrare1" / "Lucrare1" / "data",
}

QWIRKLE_TRAIN = REPO / "03_tema1-qwirkle" / "data" / "antrenare"
MOCK_DATA = PRACTICE / "data"

# Personal state (progress, cards, journal, sessions). Gitignored here; it is its
# own private repo. Override with CAVA_ME to keep it somewhere else.
ME = Path(os.environ.get("CAVA_ME", PRACTICE / "me"))


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Plug in / mount the backup drive or set CAVA_BACKUP to its root.")
    return path
