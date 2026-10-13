import numpy as np
from skimage.feature import hog
from cava_practice import synth
from cava_practice.testing import check, close, forbid


def test(m):
    forbid(m, ["skimage", "HOGDescriptor"])
    for seed, (h, w), ppc, nbins in [(0, (64, 64), 8, 9), (1, (70, 50), 6, 9), (2, (36, 36), 4, 12)]:
        img = synth.smooth_image(h, w, seed=seed).astype(np.float64)
        ref = hog(img, orientations=nbins, pixels_per_cell=(ppc, ppc), cells_per_block=(1, 1),
                  block_norm="L1", feature_vector=False).reshape(h // ppc, w // ppc, nbins)
        got = np.asarray(m.cell_histograms(img, ppc, nbins))
        check(got.shape == (h // ppc, w // ppc, nbins),
              f"cell_histograms shape {got.shape}, expected {(h // ppc, w // ppc, nbins)}")
        got_l1 = got / (got.sum(axis=-1, keepdims=True) + 1e-5)
        close(got_l1, ref, 1e-6, f"{h}x{w}, ppc={ppc}, {nbins} bins: cell histograms (compared after L1 normalising)")
        for cpb in (1, 2, 3):
            ref = hog(img, orientations=nbins, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb),
                      block_norm="L2", feature_vector=True)
            close(m.hog_descriptor(img, ppc, cpb, nbins), ref, 1e-6, f"hog_descriptor cpb={cpb}")
    edge = np.zeros((16, 16))
    edge[:, 8:] = 255
    flipped = 255 - edge
    a = np.asarray(m.cell_histograms(edge, 8, 9))
    b = np.asarray(m.cell_histograms(flipped, 8, 9))
    check(np.allclose(a, b), "orientation must be UNSIGNED (mod 180): an edge and its inverse give the same HOG")
