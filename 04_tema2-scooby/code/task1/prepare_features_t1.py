from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import cv2

from cod.Parameters import params
from cod.datasets import (
    load_gt_txt_with_prefix,
    merge_gt_dicts,
    build_image_index,
    sample_positive_crops,
    sample_negative_crops,
)
from cod.features import hog_like, hog_dim, visualize_hog


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def save_patch_grid(pos, neg, out_path: Path, grid: int = 5) -> None:
    """
    5x5: first 12 pos then 13 neg (or similar) so you see both on one image.
    """
    n = grid * grid
    items = (pos[: n // 2] + neg[: n - n // 2])

    fig, axes = plt.subplots(grid, grid, figsize=(10, 10))
    fig.suptitle("M2 patches: positives (top-left) vs negatives (rest)")

    for i in range(n):
        ax = axes[i // grid, i % grid]
        ax.axis("off")
        if i < len(items):
            rgb = items[i].crop[:, :, ::-1]
            ax.imshow(rgb)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    debug_dir = params.debug_dir
    cache_dir = params.cache_dir
    debug_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # config for this run
    NPOS = int(getattr(params, "m2_npos", 3000))  # if not in params, falls back
    NNEG = int(getattr(params, "m2_nneg", 10000))

    ok(f"dim_window={params.dim_window} cell={params.hog_cell_size} block={params.hog_block_size} bins={params.hog_num_bins}")
    ok(f"target samples: NPOS={NPOS} NNEG={NNEG}")

    train_dir = params.train_dir
    image_index = build_image_index(train_dir)
    ok(f"indexed images={len(image_index)}")

    pairs = [
        ("daphne_annotations.txt", "daphne"),
        ("fred_annotations.txt", "fred"),
        ("shaggy_annotations.txt", "shaggy"),
        ("velma_annotations.txt", "velma"),
    ]

    gt_dicts = []
    for fname, folder in pairs:
        gt = load_gt_txt_with_prefix(train_dir / fname, prefix_dir=folder)
        gt_dicts.append(gt)

    gt_all = merge_gt_dicts(gt_dicts)
    ok(f"GT(all faces): images={len(gt_all)} boxes={sum(len(v) for v in gt_all.values())}")

    # sample patches
    pos = sample_positive_crops(gt_all, image_index, n=NPOS, resize_to=params.dim_window)
    neg = sample_negative_crops(gt_all, image_index, n=NNEG, win_size=params.dim_window,
                               max_iou=params.neg_iou_thresh, resize_to=params.dim_window)
    ok(f"pos patches: N={len(pos)}")
    ok(f"neg patches: N={len(neg)}")

    # debug patch grid
    out_grid = debug_dir / "m2_patches_grid.png"
    save_patch_grid(pos, neg, out_grid, grid=5)
    ok(f"saved {out_grid}")

    # compute HOG features
    D = hog_dim(params.dim_window)
    ok(f"HOG dim: D={D}")

    pos_feat = np.zeros((len(pos), D), dtype=np.float32)
    neg_feat = np.zeros((len(neg), D), dtype=np.float32)

    for i, s in enumerate(pos):
        gray = cv2.cvtColor(s.crop, cv2.COLOR_BGR2GRAY)
        pos_feat[i] = hog_like(gray)

    for i, s in enumerate(neg):
        gray = cv2.cvtColor(s.crop, cv2.COLOR_BGR2GRAY)
        neg_feat[i] = hog_like(gray)

    ok(f"pos_features shape={pos_feat.shape} dtype={pos_feat.dtype}")
    ok(f"neg_features shape={neg_feat.shape} dtype={neg_feat.dtype}")

    # cache features + metadata
    np.save(cache_dir / "t1_pos_features.npy", pos_feat)
    np.save(cache_dir / "t1_neg_features.npy", neg_feat)
    np.save(cache_dir / "t1_pos_meta.npy", np.array([(s.file_name, s.box) for s in pos], dtype=object))
    np.save(cache_dir / "t1_neg_meta.npy", np.array([(s.file_name, s.box) for s in neg], dtype=object))

    ok(f"saved {cache_dir / 't1_pos_features.npy'}")
    ok(f"saved {cache_dir / 't1_neg_features.npy'}")

    # HOG visualizations (one example each)
    hog_pos = visualize_hog(cv2.cvtColor(pos[0].crop, cv2.COLOR_BGR2GRAY))
    hog_neg = visualize_hog(cv2.cvtColor(neg[0].crop, cv2.COLOR_BGR2GRAY))
    out_hpos = debug_dir / "m2_hog_vis_pos.png"
    out_hneg = debug_dir / "m2_hog_vis_neg.png"
    cv2.imwrite(str(out_hpos), hog_pos)
    cv2.imwrite(str(out_hneg), hog_neg)
    ok(f"saved {out_hpos}")
    ok(f"saved {out_hneg}")

    # optional: feature norm histogram (nice sanity plot)
    norms = np.concatenate([np.linalg.norm(pos_feat, axis=1), np.linalg.norm(neg_feat, axis=1)], axis=0)
    fig = plt.figure()
    plt.hist(norms, bins=50)
    plt.title("Feature L2 norm distribution (pos+neg)")
    plt.xlabel("L2 norm")
    plt.ylabel("count")
    out_norm = debug_dir / "m2_feature_norm_hist.png"
    plt.savefig(out_norm, dpi=150)
    plt.close(fig)
    ok(f"saved {out_norm}")

    ok("M2 prepare_features_t1 complete.")


if __name__ == "__main__":
    main()
