from __future__ import annotations

import sys
from pathlib import Path

# Add project root to PYTHONPATH so `import cod...` works when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../CAVA-2025-TEMA2
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA


from cod.Parameters import params
from cod.metrics import iou_xyxy
from cod.metrics import verify_feature_variability
from cod.datasets import (
    load_gt_txt_with_prefix,
    merge_gt_dicts,
    gt_box_stats,
    build_image_index,
    sample_positive_crops,
    sample_negative_crops,
)

# optional
try:
    from cod.datasets import dedup_boxes_exact
except Exception:
    dedup_boxes_exact = None


# -----------------------------
# Logging helpers
# -----------------------------
_LOG_LINES: list[str] = []

def save_pca_plot(pos_feat: np.ndarray, neg_feat: np.ndarray, out_path):
    X = np.vstack([pos_feat, neg_feat]).astype(np.float32)
    y = np.array([1]*len(pos_feat) + [0]*len(neg_feat), dtype=np.int32)

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)

    plt.figure()
    plt.scatter(Z[y==1,0], Z[y==1,1], s=5, label="pos")
    plt.scatter(Z[y==0,0], Z[y==0,1], s=5, label="neg")
    plt.legend()
    plt.title("PCA of HOG features (pos vs neg)")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] saved {out_path}")

def ok(msg: str) -> None:
    line = f"[OK] {msg}"
    print(line)
    _LOG_LINES.append(line)


def warn(msg: str) -> None:
    line = f"[WARN] {msg}"
    print(line)
    _LOG_LINES.append(line)


def save_log(debug_dir: Path, name: str = "m1_prepare_data_t1.log") -> None:
    out = debug_dir / name
    out.write_text("\n".join(_LOG_LINES) + "\n", encoding="utf-8")
    ok(f"saved log {out}")


# -----------------------------
# Debug visualizations
# -----------------------------
def save_crop_grid(sampled_crops, out_path: Path, title: str, grid: int = 5) -> None:
    n = min(len(sampled_crops), grid * grid)
    if n == 0:
        raise RuntimeError(f"No crops to visualize for: {title}")

    fig, axes = plt.subplots(grid, grid, figsize=(10, 10))
    fig.suptitle(title)

    for i in range(grid * grid):
        ax = axes[i // grid, i % grid]
        ax.axis("off")
        if i < n:
            bgr = sampled_crops[i].crop
            rgb = bgr[:, :, ::-1]  # BGR -> RGB
            ax.imshow(rgb)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_gt_size_hist(ws: np.ndarray, hs: np.ndarray, out_path: Path) -> None:
    fig = plt.figure()
    plt.hist(ws, bins=30, alpha=0.7, label="width")
    plt.hist(hs, bins=30, alpha=0.7, label="height")
    plt.legend()
    plt.title("GT face box sizes (train, merged from 4 characters)")
    plt.xlabel("pixels")
    plt.ylabel("count")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_neg_iou_hist(neg_samples, gt_all, out_path: Path, thr: float) -> None:
    """
    neg_samples: list[SampledCrop], with .file_name (rel path) and .box
    """
    max_ious = []
    for s in neg_samples:
        gt_boxes = gt_all.get(s.file_name, [])
        mi = 0.0
        for g in gt_boxes:
            mi = max(mi, iou_xyxy(s.box, g))
        max_ious.append(mi)

    max_ious = np.array(max_ious, dtype=np.float32)

    # hard verify: all negatives must be under threshold (with tiny epsilon)
    eps = 1e-6
    if len(max_ious) > 0 and (max_ious.max() > thr + eps):
        warn(f"Some negatives exceed IoU threshold! max_iou={max_ious.max():.4f} thr={thr}")
    else:
        ok(f"neg IoU check passed: max_iou={float(max_ious.max() if len(max_ious) else 0.0):.4f} thr={thr}")

    fig = plt.figure()
    plt.hist(max_ious, bins=25)
    plt.title(f"Negatives: max IoU to GT (thr={thr})")
    plt.xlabel("max IoU")
    plt.ylabel("count")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def draw_gt_overlays(
    train_dir: Path,
    image_index: dict[str, Path],
    gt_all: dict[str, list[list[int]]],
    debug_dir: Path,
    n: int = 10,
) -> None:
    """
    Saves N images with GT boxes overlaid.
    """
    keys = [k for k in gt_all.keys() if k in image_index]
    if not keys:
        warn("No GT keys found in image index for overlay.")
        return

    rng = np.random.default_rng(params.seed)
    picks = rng.choice(len(keys), size=min(n, len(keys)), replace=False)

    out_dir = debug_dir / "gt_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(picks):
        rel = keys[int(idx)]
        img_path = image_index[rel]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            warn(f"could not read overlay image: {img_path}")
            continue

        # draw boxes
        for (x1, y1, x2, y2) in gt_all[rel]:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # label
        cv2.putText(
            img, rel, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
        )

        out_path = out_dir / f"gt_overlay_{i:02d}.png"
        cv2.imwrite(str(out_path), img)

    ok(f"saved GT overlays to {out_dir}")


# -----------------------------
# Text reports
# -----------------------------
def write_filename_collisions_report(image_index: dict[str, Path], debug_dir: Path) -> None:
    """
    Reports duplicate basenames across folders (e.g. daphne/0001.jpg and fred/0001.jpg).
    """
    by_base: dict[str, list[str]] = {}
    for rel in image_index.keys():
        base = Path(rel).name
        by_base.setdefault(base, []).append(rel)

    collisions = {base: rels for base, rels in by_base.items() if len(rels) > 1}
    out = debug_dir / "filename_collisions.txt"

    lines = []
    lines.append(f"Total images indexed: {len(image_index)}")
    lines.append(f"Unique basenames: {len(by_base)}")
    lines.append(f"Colliding basenames: {len(collisions)}")
    lines.append("")
    # show first 50 collisions (enough)
    for base in sorted(collisions.keys())[:50]:
        lines.append(base)
        for rel in sorted(collisions[base]):
            lines.append(f"  - {rel}")
        lines.append("")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok(f"saved {out}")


def write_missing_images_report(gt_all: dict[str, list[list[int]]], image_index: dict[str, Path], debug_dir: Path) -> None:
    missing = [k for k in gt_all.keys() if k not in image_index]
    out = debug_dir / "gt_missing_images.txt"

    lines = []
    lines.append(f"GT images: {len(gt_all)}")
    lines.append(f"Indexed images: {len(image_index)}")
    lines.append(f"Missing GT keys in index: {len(missing)}")
    lines.append("")
    for m in missing[:200]:  # cap
        lines.append(m)

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok(f"saved {out}")


def save_params_snapshot(debug_dir: Path) -> None:
    out = debug_dir / "params_snapshot.json"
    payload = {
        "seed": params.seed,
        "train_dir": str(params.train_dir),
        "debug_dir": str(params.debug_dir),
        "dim_window": params.dim_window,
        "neg_iou_thresh": params.neg_iou_thresh,
        "step": params.step,
        "pyramid_scales": list(params.pyramid_scales),
        "hog_cell_size": params.hog_cell_size,
        "hog_block_size": params.hog_block_size,
        "hog_num_bins": params.hog_num_bins,
        "nms_iou_thresh": params.nms_iou_thresh,
        "detection_score_thresh": params.detection_score_thresh,
        "student_folder": params.student_folder,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ok(f"saved {out}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    train_dir = params.train_dir
    debug_dir = params.debug_dir
    debug_dir.mkdir(parents=True, exist_ok=True)

    ok(f"train_dir={train_dir}")
    ok(f"debug_dir={debug_dir}")

    save_params_snapshot(debug_dir)

    # 1) build image index keyed by RELATIVE PATH (e.g. 'daphne/0001.jpg')
    ok("building image index (recursive under antrenare/)...")
    image_index = build_image_index(train_dir)
    ok(f"indexed images={len(image_index)}")

    # reports about indexing
    write_filename_collisions_report(image_index, debug_dir)

    # 2) load 4 annotation files WITH PREFIX so GT keys match image_index keys
    pairs = [
        ("daphne_annotations.txt", "daphne"),
        ("fred_annotations.txt", "fred"),
        ("shaggy_annotations.txt", "shaggy"),
        ("velma_annotations.txt", "velma"),
    ]

    gt_dicts = []
    for fname, folder in pairs:
        p = train_dir / fname
        gt = load_gt_txt_with_prefix(p, prefix_dir=folder)
        num_imgs = len(gt)
        num_boxes = sum(len(v) for v in gt.values())
        ok(f"loaded {fname}: images={num_imgs}, boxes={num_boxes}")
        gt_dicts.append(gt)

    # 3) Task1 "all faces" GT = merge union of all boxes
    gt_all = merge_gt_dicts(gt_dicts)
    if dedup_boxes_exact is not None:
        gt_all = dedup_boxes_exact(gt_all)

    num_imgs_all = len(gt_all)
    num_boxes_all = sum(len(v) for v in gt_all.values())
    ok(f"merged(all faces): images={num_imgs_all}, boxes={num_boxes_all}")

    # missing GT keys report
    write_missing_images_report(gt_all, image_index, debug_dir)

    # 4) histogram of GT widths/heights
    ws, hs = gt_box_stats(gt_all)
    ok(f"w stats: min={ws.min()} med={int(np.median(ws))} max={ws.max()}")
    ok(f"h stats: min={hs.min()} med={int(np.median(hs))} max={hs.max()}")

    out_hist = debug_dir / "gt_face_wh_hist.png"
    save_gt_size_hist(ws, hs, out_hist)
    ok(f"saved {out_hist}")

    # 5) sample crops
    pos = sample_positive_crops(
        gt_all,
        image_index,
        n=25,
        resize_to=params.dim_window,
    )

    neg = sample_negative_crops(
        gt_all,
        image_index,
        n=25,
        win_size=params.dim_window,
        max_iou=params.neg_iou_thresh,
        resize_to=params.dim_window,
    )

    ok(f"sampled crops: pos={len(pos)}, neg={len(neg)}")

    # 6) save grids
    out_pos = debug_dir / "pos_grid.png"
    out_neg = debug_dir / "neg_grid.png"
    save_crop_grid(pos, out_pos, "Positive crops (faces)", grid=5)
    save_crop_grid(neg, out_neg, "Negative crops (background)", grid=5)
    ok(f"saved {out_pos}")
    ok(f"saved {out_neg}")

    # 7) GT overlay images
    draw_gt_overlays(train_dir, image_index, gt_all, debug_dir, n=10)

    # 8) Neg IoU histogram
    out_neg_iou = debug_dir / "neg_max_iou_hist.png"
    save_neg_iou_hist(neg, gt_all, out_neg_iou, thr=params.neg_iou_thresh)
    ok(f"saved {out_neg_iou}")

    ok("M1 prepare_data_t1 complete.")
    save_log(debug_dir)


if __name__ == "__main__":
    main()
