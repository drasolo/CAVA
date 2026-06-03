from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


import cv2
import numpy as np

from cod.Parameters import params
from cod.metrics import iou_xyxy


Box = List[int]  # [x1, y1, x2, y2]
GTDict = Dict[str, List[Box]]  # filename -> list of boxes


@dataclass(frozen=True)
class SampledCrop:
    file_name: str
    box: Box
    crop: np.ndarray  # HxWxC (BGR from cv2)


def load_gt_txt_with_prefix(gt_path: Path, prefix_dir: str) -> GTDict:
    """
    Reads GT file (filename x1 y1 x2 y2) and converts keys to 'prefix_dir/filename'
    Example: '0694.jpg' -> 'daphne/0694.jpg'
    This does NOT rename any files on disk.
    """
    raw = load_gt_txt(gt_path)  # your existing loader: filename -> boxes
    out: GTDict = {}
    for fn, boxes in raw.items():
        out[f"{prefix_dir}/{fn}"] = boxes
    return out



def read_image_bgr(image_index: dict[str, Path], rel_path: str) -> np.ndarray:
    img_path = image_index.get(rel_path)
    if img_path is None:
        raise RuntimeError(f"Image not found in index: {rel_path}")
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    return img



def clip_box_xyxy(box: Box, w: int, h: int) -> Box:
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return [x1, y1, x2, y2]


def safe_crop(img: np.ndarray, box: Box) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = clip_box_xyxy(box, w, h)
    if x2 < x1 or y2 < y1:
        return None
    crop = img[y1:y2 + 1, x1:x2 + 1]
    if crop.size == 0:
        return None
    return crop


def build_image_index(images_root: Path) -> dict[str, Path]:
    """
    Index images by RELATIVE path (e.g. 'daphne/0001.jpg'), not by filename.
    This avoids collisions when multiple folders contain '0001.jpg'.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    index: dict[str, Path] = {}

    for p in images_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(images_root).as_posix()  # e.g. daphne/0001.jpg
        index[rel] = p

    if not index:
        raise RuntimeError(f"No images found under: {images_root}")
    return index



def load_gt_txt(gt_path: Path) -> GTDict:
    """
    GT file format: filename x1 y1 x2 y2
    Returns: dict[filename] -> list of boxes.
    """
    if not gt_path.exists():
        raise RuntimeError(f"GT file not found: {gt_path}")

    data = np.loadtxt(str(gt_path), dtype=str)
    if data.ndim == 1:
        data = data[None, :]  # single-line file

    gt: GTDict = {}
    for row in data:
        file_name = row[0]
        box = [int(row[1]), int(row[2]), int(row[3]), int(row[4])]
        gt.setdefault(file_name, []).append(box)

    return gt
    
def load_gt_txt_with_prefix(gt_path: Path, prefix_dir: str) -> GTDict:
    """
    Reads GT (filename x1 y1 x2 y2) and rewrites keys to 'prefix_dir/filename'
    Example: prefix_dir='daphne' => 'daphne/0001.jpg'
    DOES NOT rename files on disk.
    """
    raw = load_gt_txt(gt_path)  # your existing loader: filename -> boxes
    out: GTDict = {}
    for fn, boxes in raw.items():
        out[f"{prefix_dir}/{fn}"] = boxes
    return out


def gt_box_stats(gt: GTDict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns arrays of widths and heights for all GT boxes.
    Uses +1 convention to match Pascal/VOC-style inclusive coords.
    """
    ws = []
    hs = []
    for _, boxes in gt.items():
        for (x1, y1, x2, y2) in boxes:
            ws.append(x2 - x1 + 1)
            hs.append(y2 - y1 + 1)
    return np.array(ws, dtype=np.int32), np.array(hs, dtype=np.int32)


def list_images(images_dir: Path) -> List[str]:
    """
    Lists image filenames in a directory (common extensions).
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p.name for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def sample_positive_crops(
    gt: GTDict,
    images_index: Path,
    n: int = 25,
    resize_to: Optional[int] = None,
) -> List[SampledCrop]:
    """
    Randomly samples GT boxes and returns crops.
    """
    rng = np.random.default_rng(params.seed)

    all_items: List[Tuple[str, Box]] = []
    for fn, boxes in gt.items():
        for b in boxes:
            all_items.append((fn, b))

    if len(all_items) == 0:
        raise RuntimeError("No GT boxes found for positives.")

    idxs = rng.choice(len(all_items), size=min(n, len(all_items)), replace=False)

    out: List[SampledCrop] = []
    for i in idxs:
        fn, box = all_items[i]
        img = read_image_bgr(images_index, fn)
        crop = safe_crop(img, box)
        if crop is None:
            continue
        if resize_to is not None:
            crop = cv2.resize(crop, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
        out.append(SampledCrop(file_name=fn, box=box, crop=crop))

    return out


def merge_gt_dicts(dicts: list[GTDict]) -> GTDict:
    """
    Merge multiple GT dicts (filename -> list of boxes) into one.
    Used to build Task1 "all faces" GT from 4 character annotation files.
    """
    merged: GTDict = {}
    for d in dicts:
        for fn, boxes in d.items():
            merged.setdefault(fn, []).extend(boxes)
    return merged


def dedup_boxes_exact(gt: GTDict) -> GTDict:
    """
    Remove exact duplicate boxes per image (same coords).
    """
    out: GTDict = {}
    for fn, boxes in gt.items():
        seen = set()
        unique = []
        for b in boxes:
            t = tuple(map(int, b))
            if t not in seen:
                seen.add(t)
                unique.append(list(t))
        out[fn] = unique
    return out


def sample_negative_crops(
    gt: GTDict,
    images_index: Path,
    n: int = 25,
    win_size: Optional[int] = None,
    max_iou: Optional[float] = None,
    max_tries_per_crop: int = 200,
    resize_to: Optional[int] = None,
) -> List[SampledCrop]:
    """
    Samples random windows that do NOT overlap GT faces.
    A window is accepted if max IoU with any GT on that image < max_iou.
    """
    rng = np.random.default_rng(params.seed)

    win = win_size if win_size is not None else params.dim_window
    thr = max_iou if max_iou is not None else params.neg_iou_thresh

    file_names = list(gt.keys())
    if len(file_names) == 0:
        raise RuntimeError("GT dict is empty; cannot sample negatives by image.")

    out: List[SampledCrop] = []

    while len(out) < n:
        fn = file_names[int(rng.integers(0, len(file_names)))]
        img = read_image_bgr(images_index, fn)
        h, w = img.shape[:2]

        if w < win or h < win:
            # skip tiny images
            continue

        gt_boxes = gt[fn]

        accepted = False
        for _ in range(max_tries_per_crop):
            x1 = int(rng.integers(0, w - win + 1))
            y1 = int(rng.integers(0, h - win + 1))
            box = [x1, y1, x1 + win - 1, y1 + win - 1]

            # compute max IoU with any GT box
            ious = [iou_xyxy(box, gt_box) for gt_box in gt_boxes]
            if (max(ious) if len(ious) else 0.0) < thr:
                crop = safe_crop(img, box)
                if crop is None:
                    continue
                if resize_to is not None:
                    crop = cv2.resize(crop, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
                out.append(SampledCrop(file_name=fn, box=box, crop=crop))
                accepted = True
                break

        if not accepted:
            # if we fail to find a good negative in this image, try another image
            continue

    return out
