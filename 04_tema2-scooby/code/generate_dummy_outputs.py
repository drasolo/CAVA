import os
from pathlib import Path
import numpy as np


def load_env_file(env_path: Path) -> None:
    """
    Minimal .env loader (no external deps).
    Supports lines like: KEY=value
    Ignores empty lines and comments (# ...).
    """
    if not env_path.exists():
        raise RuntimeError(f"Missing .env file at: {env_path}")

    with env_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # don't override already-set environment variables
            os.environ.setdefault(key, value)


def get_student_folder() -> str:
    project_root = Path(__file__).resolve().parents[1]  # parent of cod/
    env_path = project_root / ".env"
    load_env_file(env_path)

    student_folder = os.getenv("STUDENT_FOLDER", "").strip()
    if not student_folder:
        raise RuntimeError("Missing STUDENT_FOLDER in .env (e.g. STUDENT_FOLDER=331_Alexe_Bogdan)")
    return student_folder


def save_task1(out_task1_dir: Path) -> None:
    out_task1_dir.mkdir(parents=True, exist_ok=True)

    detections = np.empty((0, 4), dtype=np.int32)      # Nx4 [xmin, ymin, xmax, ymax]
    scores = np.empty((0,), dtype=np.float32)          # N
    file_names = np.empty((0,), dtype=object)          # N strings (object safest)

    np.save(out_task1_dir / "detections_all_faces.npy", detections)
    np.save(out_task1_dir / "scores_all_faces.npy", scores)
    np.save(out_task1_dir / "file_names_all_faces.npy", file_names)

    print("[OK] Task1 dummy saved")
    print("  path:", out_task1_dir)
    print("  detections:", detections.shape, detections.dtype)
    print("  scores:    ", scores.shape, scores.dtype)
    print("  file_names:", file_names.shape, file_names.dtype)


def save_task2(out_task2_dir: Path) -> None:
    out_task2_dir.mkdir(parents=True, exist_ok=True)

    characters = ["daphne", "fred", "shaggy", "velma"]
    for c in characters:
        detections = np.empty((0, 4), dtype=np.int32)
        scores = np.empty((0,), dtype=np.float32)
        file_names = np.empty((0,), dtype=object)

        np.save(out_task2_dir / f"detections_{c}.npy", detections)
        np.save(out_task2_dir / f"scores_{c}.npy", scores)
        np.save(out_task2_dir / f"file_names_{c}.npy", file_names)

    print("[OK] Task2 dummy saved for:", ", ".join(characters))
    print("  path:", out_task2_dir)


def quick_self_check(out_task1_dir: Path, out_task2_dir: Path) -> None:
    # Task 1 checks
    d = np.load(out_task1_dir / "detections_all_faces.npy", allow_pickle=True)
    s = np.load(out_task1_dir / "scores_all_faces.npy", allow_pickle=True)
    f = np.load(out_task1_dir / "file_names_all_faces.npy", allow_pickle=True)

    assert d.ndim == 2 and d.shape[1] == 4, "Task1 detections must be Nx4"
    assert s.ndim == 1 and f.ndim == 1, "Task1 scores and file_names must be 1D"
    assert len(d) == len(s) == len(f), "Task1 arrays must have same length"
    print("[OK] Task1 self-check passed")

    # Task 2 checks
    for c in ["daphne", "fred", "shaggy", "velma"]:
        d = np.load(out_task2_dir / f"detections_{c}.npy", allow_pickle=True)
        s = np.load(out_task2_dir / f"scores_{c}.npy", allow_pickle=True)
        f = np.load(out_task2_dir / f"file_names_{c}.npy", allow_pickle=True)

        assert d.ndim == 2 and d.shape[1] == 4, f"Task2 {c}: detections must be Nx4"
        assert s.ndim == 1 and f.ndim == 1, f"Task2 {c}: scores and file_names must be 1D"
        assert len(d) == len(s) == len(f), f"Task2 {c}: arrays must have same length"

    print("[OK] Task2 self-check passed")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]  # parent of cod/
    student_folder = get_student_folder()

    out_task1_dir = project_root / "evaluare" / "fisiere_solutie" / student_folder / "task1"
    out_task2_dir = project_root / "evaluare" / "fisiere_solutie" / student_folder / "task2"

    save_task1(out_task1_dir)
    save_task2(out_task2_dir)
    quick_self_check(out_task1_dir, out_task2_dir)

    print("[DONE] Dummy outputs ready. Next: run the evaluator.")


if __name__ == "__main__":
    main()
