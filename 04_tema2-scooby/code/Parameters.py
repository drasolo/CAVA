from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


# -----------------------------
# Project root + .env loading
# -----------------------------
def _project_root() -> Path:
    # cod/Parameters.py -> project root is parent of cod/
    return Path(__file__).resolve().parents[1]


def _load_env_file(env_path: Path) -> None:
    """
    Minimal .env loader.
    Supports lines: KEY=value
    Ignores empty lines and comments (# ...).
    Does NOT override already-set environment variables.
    """
    if not env_path.exists():
        raise RuntimeError(f"Missing .env file at: {env_path}")

    with env_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)


def _env_str(key: str, default: Optional[str] = None) -> str:
    v = os.getenv(key, default if default is not None else "")
    return v.strip()


def _env_int(key: str, default: int) -> int:
    v = _env_str(key, str(default))
    try:
        return int(v)
    except ValueError:
        raise RuntimeError(f"Env var {key} must be int, got: {v!r}")


def _env_float(key: str, default: float) -> float:
    v = _env_str(key, str(default))
    try:
        return float(v)
    except ValueError:
        raise RuntimeError(f"Env var {key} must be float, got: {v!r}")


def _env_bool(key: str, default: bool) -> bool:
    v = _env_str(key, "1" if default else "0").lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise RuntimeError(f"Env var {key} must be boolean-like, got: {v!r}")


def _env_list_float(key: str, default: List[float]) -> List[float]:
    """
    Reads comma-separated floats from env.
    Example: PYRAMID_SCALES=1.0,0.9,0.8,0.7
    """
    raw = _env_str(key, "")
    if not raw:
        return default
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    try:
        return [float(p) for p in parts]
    except ValueError:
        raise RuntimeError(f"Env var {key} must be comma-separated floats, got: {raw!r}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------------
# Params dataclass
# -----------------------------
@dataclass(frozen=True)
class Params:
    # Core paths
    root_dir: Path
    cod_dir: Path

    train_dir: Path
    val_dir: Path
    test_dir: Path

    debug_dir: Path
    cache_dir: Path
    models_dir: Path

    # Evaluator output paths
    student_folder: str
    solution_root: Path
    task1_out_dir: Path
    task2_out_dir: Path

    # Reproducibility
    seed: int

    # Detector params
    dim_window: int
    step: int
    pyramid_scales: List[float]

    # HOG-like feature params
    hog_cell_size: int
    hog_block_size: int
    hog_num_bins: int

    # Thresholds / post-processing
    nms_iou_thresh: float
    detection_score_thresh: float
    topk_pre_nms: int

    # Sampling thresholds
    neg_iou_thresh: float

    # Behavior toggles
    save_debug: bool


def build_params() -> Params:
    root = _project_root()
    _load_env_file(root / ".env")

    # Base dirs (from .env or defaults)
    train_dir = root / _env_str("TRAIN_DIR", "antrenare")
    val_dir = root / _env_str("VAL_DIR", "validare")
    test_dir = root / _env_str("TEST_DIR", "testare")

    debug_dir = root / _env_str("DEBUG_DIR", "debug")
    cache_dir = root / _env_str("CACHE_DIR", "cache")
    models_dir = root / _env_str("MODELS_DIR", "models")

    # Evaluator outputs
    student_folder = _env_str("STUDENT_FOLDER", "")
    if not student_folder:
        raise RuntimeError("Missing STUDENT_FOLDER in .env (e.g. STUDENT_FOLDER=331_Alexe_Bogdan)")

    solution_root = root / "evaluare" / "fisiere_solutie" / student_folder
    task1_run_subdir = _env_str("TASK1_RUN_SUBDIR", "task1")
    task1_out_dir = solution_root / task1_run_subdir
    task2_out_dir = solution_root / "task2"

    # Repro
    seed = _env_int("SEED", 42)

    # Detector params (env overridable)
    dim_window = _env_int("DIM_WINDOW", 64)
    step = _env_int("STEP", 6)
    pyramid_scales = _env_list_float("PYRAMID_SCALES", [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6])

    hog_cell_size = _env_int("HOG_CELL_SIZE", 8)
    hog_block_size = _env_int("HOG_BLOCK_SIZE", 2)
    hog_num_bins = _env_int("HOG_NUM_BINS", 9)

    nms_iou_thresh = _env_float("NMS_IOU_THRESH", 0.15)
    detection_score_thresh = _env_float("DETECTION_SCORE_THRESH", 0.0)

    # NEW: cap candidates before NMS
    topk_pre_nms = _env_int("TOPK_PRE_NMS", 300)

    neg_iou_thresh = _env_float("NEG_IOU_THRESH", 0.1)

    save_debug = _env_bool("SAVE_DEBUG", True)

    # Create commonly used dirs (safe)
    ensure_dir(debug_dir)
    ensure_dir(cache_dir)
    ensure_dir(models_dir)
    ensure_dir(task1_out_dir)
    ensure_dir(task2_out_dir)

    # Seed global RNGs early
    set_global_seed(seed)

    return Params(
        root_dir=root,
        cod_dir=root / "cod",
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        debug_dir=debug_dir,
        cache_dir=cache_dir,
        models_dir=models_dir,
        student_folder=student_folder,
        solution_root=solution_root,
        task1_out_dir=task1_out_dir,
        task2_out_dir=task2_out_dir,
        seed=seed,
        dim_window=dim_window,
        step=step,
        pyramid_scales=pyramid_scales,
        hog_cell_size=hog_cell_size,
        hog_block_size=hog_block_size,
        hog_num_bins=hog_num_bins,
        nms_iou_thresh=nms_iou_thresh,
        detection_score_thresh=detection_score_thresh,
        topk_pre_nms=topk_pre_nms,
        neg_iou_thresh=neg_iou_thresh,
        save_debug=save_debug,
    )


# Global singleton-style params (import-friendly)
params: Params = build_params()


def reload_params() -> Params:
    """
    If you edit .env during a session and want to reload without restarting Python.
    """
    global params
    params = build_params()
    return params


def print_params(p: Params = params) -> None:
    print("[OK] Params loaded")
    print(" root_dir:", p.root_dir)
    print(" train_dir:", p.train_dir)
    print(" val_dir:", p.val_dir)
    print(" test_dir:", p.test_dir)
    print(" debug_dir:", p.debug_dir)
    print(" cache_dir:", p.cache_dir)
    print(" models_dir:", p.models_dir)
    print(" student_folder:", p.student_folder)
    print(" task1_out_dir:", p.task1_out_dir)
    print(" task2_out_dir:", p.task2_out_dir)
    print(" seed:", p.seed)
    print(" dim_window:", p.dim_window)
    print(" step:", p.step)
    print(" pyramid_scales:", p.pyramid_scales)
    print(" hog: cell/block/bins:", p.hog_cell_size, p.hog_block_size, p.hog_num_bins)
    print(" detection_score_thresh:", p.detection_score_thresh)
    print(" topk_pre_nms:", p.topk_pre_nms)
    print(" nms_iou_thresh:", p.nms_iou_thresh)
    print(" neg_iou_thresh:", p.neg_iou_thresh)
    print(" save_debug:", p.save_debug)
