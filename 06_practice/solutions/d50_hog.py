import numpy as np


def cell_histograms(gray, ppc, nbins):
    I = np.asarray(gray, np.float64)
    gr = np.zeros_like(I)
    gc = np.zeros_like(I)
    gr[1:-1, :] = I[2:, :] - I[:-2, :]
    gc[:, 1:-1] = I[:, 2:] - I[:, :-2]
    mag = np.hypot(gc, gr)
    ori = np.rad2deg(np.arctan2(gr, gc)) % 180
    ny, nx = I.shape[0] // ppc, I.shape[1] // ppc
    mag = mag[:ny * ppc, :nx * ppc]
    ori = ori[:ny * ppc, :nx * ppc]
    width = 180.0 / nbins
    out = np.zeros((ny, nx, nbins))
    for i in range(nbins):
        sel = (ori >= i * width) & (ori < (i + 1) * width)
        m = np.where(sel, mag, 0).reshape(ny, ppc, nx, ppc)
        out[..., i] = m.sum(axis=(1, 3)) / (ppc * ppc)
    return out


def hog_descriptor(gray, ppc, cpb, nbins):
    h = cell_histograms(gray, ppc, nbins)
    ny, nx, _ = h.shape
    blocks = []
    for y in range(ny - cpb + 1):
        for x in range(nx - cpb + 1):
            v = h[y:y + cpb, x:x + cpb].ravel()
            blocks.append(v / np.sqrt(np.sum(v ** 2) + 1e-10))
    return np.concatenate(blocks)
