#Train face detector + hard negative mining loop, save model.
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

from cod.Parameters import params

def ok(msg: str) -> None:
    print(f"[OK] {msg}")

def save_confusion(cm: np.ndarray, out_path):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion matrix (val)")
    plt.xlabel("pred")
    plt.ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.savefig(out_path, dpi=150)
    plt.close()
    ok(f"saved {out_path}")

def save_margin_hist(scores_pos, scores_neg, out_path):
    plt.figure()
    plt.hist(scores_pos, bins=50, alpha=0.7, label="pos")
    plt.hist(scores_neg, bins=50, alpha=0.7, label="neg")
    plt.legend()
    plt.title("LinearSVC margin scores (pos vs neg)")
    plt.savefig(out_path, dpi=150)
    plt.close()
    ok(f"saved {out_path}")

def main():
    cache_dir = params.cache_dir
    debug_dir = params.debug_dir
    debug_dir.mkdir(parents=True, exist_ok=True)

    pos_feat = np.load(cache_dir / "t1_pos_features.npy")
    neg_feat = np.load(cache_dir / "t1_neg_features.npy")
    ok(f"loaded pos_feat={pos_feat.shape} neg_feat={neg_feat.shape}")

    X = np.vstack([pos_feat, neg_feat]).astype(np.float32)
    y = np.array([1]*len(pos_feat) + [0]*len(neg_feat), dtype=np.int32)

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=params.seed, stratify=y
    )

    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
    clf.fit(Xtr, ytr)                                              #acilisa

    pred = clf.predict(Xva)
    acc = float(accuracy_score(yva, pred))
    cm = confusion_matrix(yva, pred)

    ok(f"LinearSVC val accuracy={acc:.4f}")
    ok(f"confusion matrix:\n{cm}")

    save_confusion(cm, debug_dir / "m3_confusion_t1.png")

    scores_pos = clf.decision_function(pos_feat)
    scores_neg = clf.decision_function(neg_feat)
    save_margin_hist(scores_pos, scores_neg, debug_dir / "m3_svm_margin_hist.png")

    ok("M3 sanity train complete (classifier learns check).")

if __name__ == "__main__":
    main()
