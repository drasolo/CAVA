import cv2
import numpy as np
from cava_practice.testing import check, forbid


def same_partition(a, b):
    pairs = np.unique(np.stack([a.ravel(), b.ravel()]), axis=1)
    return pairs.shape[1] == len(np.unique(a)) == len(np.unique(b))


def test(m):
    forbid(m, ["connectedComponents", "connectedComponentsWithStats", "skimage", "scipy"])
    diag = np.array([[1, 0], [0, 1]], np.uint8)
    n4, _ = m.label_components(diag, 4)
    n8, _ = m.label_components(diag, 8)
    check((n4, n8) == (2, 1), f"two pixels touching at a corner: expected 2 components (4-conn) and 1 (8-conn), got {n4} and {n8}")
    r = np.random.default_rng(2)
    for trial in range(3):
        b = np.where(r.random((30, 40)) > 0.55, 255, 0).astype(np.uint8)
        for conn in (4, 8):
            ref_n, ref = cv2.connectedComponents(b, connectivity=conn)
            n, lab = m.label_components(b, conn)
            lab = np.asarray(lab)
            check(n == ref_n - 1, f"{conn}-connectivity: found {n} components, expected {ref_n - 1} (background is not a component)")
            check(lab.shape == b.shape and (lab[b == 0] == 0).all(), "background pixels must have label 0")
            check(same_partition(lab, ref), f"{conn}-connectivity: pixels are grouped differently from OpenCV")
            areas = m.component_areas(lab, n)
            check(sorted(areas) == sorted(np.bincount(ref.ravel())[1:].tolist()), "component_areas don't match")
