# cod/task1/run_t1.py
# Run detector (Task1) on debug/val/test and (for val/test) write evaluator .npy outputs.
#
# Fixes included:
# - DEBUG_MAX_IMAGES works for debug/val/test (fast sanity runs)
# - SAVE_DEBUG and SAVE_SCORE_HISTS env flags actually control output
# - Matplotlib forced to Agg backend (safe headless runs)
# - Optional mock IO mode (T1_MOCK_IO=1) to validate read/write paths fast

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2

# Force non-interactive backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from sklearn.svm import LinearSVC

# Make `import cod...` work when running directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cod.Parameters import params
from cod.features import hog_like
from cod.metrics import nms_xyxy


def ok(msg: str) -> None:
    print(f"[OK] {msg}", flush=True)


def env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    v = (v or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def save_score_hist(scores: np.ndarray, out_path: Path) -> None:
    plt.figure()
    plt.hist(scores, bins=80)
    plt.title("Scores of kept windows (after threshold)")
    plt.xlabel("SVM score")
    plt.ylabel("count")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def draw_boxes(
    img_bgr: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray | None,
    color: tuple[int, int, int] = (0, 255, 0),
    topk: int = 200,
) -> np.ndarray:
    out = img_bgr.copy()
    if boxes is None or len(boxes) == 0:
        return out

    k = min(len(boxes), topk)
    for i in range(k):
        x1, y1, x2, y2 = boxes[i].astype(int).tolist()
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if scores is not None and i < len(scores):
            cv2.putText(
                out,
                f"{float(scores[i]):.2f}",
                (x1, max(0, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return out


def image_pyramid(img: np.ndarray, scales: list[float]):
    """Yields (scale, resized_img). scale is relative to original image."""
    h0, w0 = img.shape[:2]
    for s in scales:
        if s <= 0:
            continue
        w = int(round(w0 * s))
        h = int(round(h0 * s))
        if w < params.dim_window or h < params.dim_window:
            continue
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        yield s, resized


def sliding_windows(img: np.ndarray, win: int, step: int):
    H, W = img.shape[:2]
    for y in range(0, H - win + 1, step):
        for x in range(0, W - win + 1, step):
            yield x, y, img[y : y + win, x : x + win]


def train_or_load_detector() -> LinearSVC:
    """
    Baseline: train from cached features each time.
    """
    cache_dir = params.cache_dir
    pos_path = cache_dir / "t1_pos_features.npy"
    neg_path = cache_dir / "t1_neg_features.npy"

    if not pos_path.exists() or not neg_path.exists():
        raise RuntimeError(
            "Missing cached features. Run prepare_data_t1.py first to generate:\n"
            f"  {pos_path}\n"
            f"  {neg_path}"
        )

    pos_feat = np.load(pos_path)
    neg_feat = np.load(neg_path)
    ok(f"loaded pos_feat={pos_feat.shape} neg_feat={neg_feat.shape}")

    X = np.vstack([pos_feat, neg_feat]).astype(np.float32)
    y = np.concatenate(
        [np.ones(len(pos_feat), dtype=np.int32), np.zeros(len(neg_feat), dtype=np.int32)]
    )

    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
    clf.fit(X, y)
    ok("trained LinearSVC detector")
    return clf


def build_image_index_recursive(root_dir: Path) -> dict[str, Path]:
    """
    Index images by *relative path from root_dir* (e.g. 'daphne/0001.jpg').
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    index: dict[str, Path] = {}
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            rel = p.relative_to(root_dir).as_posix()
            index[rel] = p
    return index


def save_task1_npy(dets_xyxy: np.ndarray, scores: np.ndarray, file_names: np.ndarray) -> None:
    """
    Save evaluator-format Task1 files into:
      evaluare/fisiere_solutie/<STUDENT_FOLDER>/<TASK1_RUN_SUBDIR>/
    i.e. params.task1_out_dir (already includes TASK1_RUN_SUBDIR).
    """
    out_dir = params.task1_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dets_xyxy = np.asarray(dets_xyxy, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)

    # evaluator wants plain python strings; keep basenames ("0001.jpg")
    file_names = np.asarray(file_names, dtype=str)

    np.save(out_dir / "detections_all_faces.npy", dets_xyxy)
    np.save(out_dir / "scores_all_faces.npy", scores)
    np.save(out_dir / "file_names_all_faces.npy", file_names)

    ok("[Task1] saved .npy outputs")
    ok(f"  path: {out_dir}")
    ok(f"  detections: {dets_xyxy.shape} int32")
    ok(f"  scores:     {scores.shape} float32")
    ok(f"  file_names: {file_names.shape} str")


def mock_io_write() -> None:
    """
    Fast mode to validate read/write paths without scanning windows.
    Enable with: T1_MOCK_IO=1
    Writes a tiny valid output set to params.task1_out_dir and exits.
    """
    out_dir = params.task1_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dets = np.array([[0, 0, 63, 63]], dtype=np.int32)
    scr = np.array([1.234], dtype=np.float32)
    fns = np.array(["0001.jpg"], dtype=str)

    np.save(out_dir / "detections_all_faces.npy", dets)
    np.save(out_dir / "scores_all_faces.npy", scr)
    np.save(out_dir / "file_names_all_faces.npy", fns)

    ok("[MOCK] wrote minimal npy outputs (1 detection) to validate paths")
    ok(f"[MOCK] out_dir={out_dir}")


def main() -> None:
    # Choose split:
    # RUN_SPLIT=debug|val|test
    run_split = os.getenv("RUN_SPLIT", "").strip().lower()
    if not run_split:
        # If you later add it to Params, this will still work.
        run_split = getattr(params, "run_split", "debug")

    run_split = str(run_split).strip().lower()
    if run_split not in {"debug", "val", "test"}:
        raise RuntimeError(f"RUN_SPLIT must be one of: debug|val|test. Got: {run_split!r}")

    # Debug output controls
    save_debug = _env_flag("SAVE_DEBUG", "1")
    save_score_hists = _env_flag("SAVE_SCORE_HISTS", "0")
    max_images = int((os.getenv("DEBUG_MAX_IMAGES", "0") or "0").strip() or "0")
    mock_io = _env_flag("T1_MOCK_IO", "0")

    # Where to scan
    if run_split == "debug":
        scan_dir = params.train_dir
    elif run_split == "val":
        scan_dir = params.val_dir
    else:
        scan_dir = params.test_dir

    debug_out = params.debug_dir / "t1_run"
    debug_out.mkdir(parents=True, exist_ok=True)

    # Mock mode: validate output folder & evaluator format quickly
    if mock_io and run_split in {"val", "test"}:
        ok(f"RUN_SPLIT={run_split} scan_dir={scan_dir}")
        ok("T1_MOCK_IO=1 enabled -> skipping scan, writing minimal outputs.")
        mock_io_write()
        ok("Next: run evaluator (for val) to get PR/AP.")
        return

    clf = train_or_load_detector()

    image_index = build_image_index_recursive(scan_dir)
    keys = sorted(image_index.keys())

    # Optional hard cap (works for debug/val/test)
    max_images = env_int("DEBUG_MAX_IMAGES", 0)

    # Default debug cap if user didn't specify one
    if run_split == "debug" and max_images <= 0:
        max_images = 5

    # Apply cap if enabled
    if max_images > 0:
        keys = keys[:max_images]


    ok(f"RUN_SPLIT={run_split} scan_dir={scan_dir}")
    ok(f"images indexed={len(image_index)} will_process={len(keys)}")

    win = params.dim_window
    step = params.step
    score_thr = params.detection_score_thresh
    nms_thr = params.nms_iou_thresh
    scales = list(params.pyramid_scales)
    topk_pre = params.topk_pre_nms

    ok(f"win={win} step={step} score_thr={score_thr} nms_thr={nms_thr} topk_pre_nms={topk_pre}")
    ok(f"pyramid scales={scales}")

    out_boxes: list[list[int]] = []
    out_scores: list[float] = []
    out_names: list[str] = []

    for idx, rel in enumerate(keys, start=1):
        img_path = image_index[rel]
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            ok(f"[{idx}] skip unreadable: {img_path}")
            continue

        t0 = time.time()

        cand_boxes: list[list[int]] = []
        cand_scores: list[float] = []

        # multiscale scanning
        for s, im_s in image_pyramid(img_bgr, scales):
            for x, y, patch in sliding_windows(im_s, win, step):
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                feat = hog_like(gray).reshape(1, -1)
                score = float(clf.decision_function(feat)[0])
                if score < score_thr:
                    continue

                # map box back to original image coordinates
                x1 = int(round(x / s))
                y1 = int(round(y / s))
                x2 = int(round((x + win - 1) / s))
                y2 = int(round((y + win - 1) / s))
                cand_boxes.append([x1, y1, x2, y2])
                cand_scores.append(score)

        all_boxes = np.array(cand_boxes, dtype=np.int32) if cand_boxes else np.zeros((0, 4), dtype=np.int32)
        all_scores = np.array(cand_scores, dtype=np.float32) if cand_scores else np.zeros((0,), dtype=np.float32)

        ok(f"[{idx}] {rel}: det_pre_nms={len(all_boxes)}")

        # sort + stats + optional histogram + topK cap (pre-NMS)
        if len(all_scores) > 0:
            order = np.argsort(all_scores)[::-1]
            all_boxes = all_boxes[order]
            all_scores = all_scores[order]

            mn = float(all_scores.min())
            md = float(np.median(all_scores))
            mx = float(all_scores.max())
            ok(f"[{idx}] score stats: min={mn:.3f} med={md:.3f} max={mx:.3f} thr={score_thr:.3f}")

            if save_score_hists and (run_split == "debug" or idx <= 10):
                save_score_hist(all_scores, debug_out / f"score_hist_{idx:04d}.png")
                ok(f"[{idx}] saved: score_hist_{idx:04d}.png")

            if len(all_scores) > topk_pre:
                all_boxes = all_boxes[:topk_pre]
                all_scores = all_scores[:topk_pre]
            ok(f"[{idx}] kept topK_pre_nms={min(topk_pre, len(all_scores))}")
        else:
            ok(f"[{idx}] score stats: EMPTY thr={score_thr:.3f}")

        # debug overlays
        if save_debug and (run_split == "debug" or idx <= 10):
            before = draw_boxes(img_bgr, all_boxes, all_scores, color=(0, 0, 255), topk=200)
            cv2.imwrite(str(debug_out / f"det_before_nms_{idx:04d}.png"), before)

        # NMS
        keep = nms_xyxy(all_boxes, all_scores, iou_thr=nms_thr)
        boxes_nms = all_boxes[keep] if len(keep) else np.zeros((0, 4), dtype=np.int32)
        scores_nms = all_scores[keep] if len(keep) else np.zeros((0,), dtype=np.float32)

        ok(f"[{idx}] {rel}: det_post_nms={len(boxes_nms)}")
        top = scores_nms[:10] if len(scores_nms) else np.array([], dtype=np.float32)
        ok(f"[{idx}] top post_nms scores: {np.round(top, 2)}")

        if save_debug and (run_split == "debug" or idx <= 10):
            after = draw_boxes(img_bgr, boxes_nms, scores_nms, color=(0, 255, 0), topk=200)
            cv2.imwrite(str(debug_out / f"det_after_nms_{idx:04d}.png"), after)

        ok(f"[{idx}] time={time.time() - t0:.2f}s")

        # IMPORTANT: evaluator expects basenames like "0001.jpg"
        if run_split in {"val", "test"} and len(boxes_nms) > 0:
            base = img_path.name  # robust: uses the actual path name, not rel
            for b, s in zip(boxes_nms.tolist(), scores_nms.tolist()):
                out_boxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
                out_scores.append(float(s))
                out_names.append(base)

    ok("scan done.")

    if run_split in {"val", "test"}:
        dets = np.array(out_boxes, dtype=np.int32) if out_boxes else np.zeros((0, 4), dtype=np.int32)
        scr = np.array(out_scores, dtype=np.float32) if out_scores else np.zeros((0,), dtype=np.float32)
        fns = np.array(out_names, dtype=str) if out_names else np.zeros((0,), dtype=str)
        save_task1_npy(dets, scr, fns)
        ok("Next: run evaluator (for val) to get PR/AP.")
    else:
        ok("Debug run complete. Check debug/t1_run/ images.")


if __name__ == "__main__":
    main()
