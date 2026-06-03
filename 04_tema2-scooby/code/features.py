from __future__ import annotations

import numpy as np
import cv2

from cod.Parameters import params


# -----------------------------
# Core HOG-like extractor
# -----------------------------
def _to_gray_u8(gray: np.ndarray) -> np.ndarray:
    """
    Accepts grayscale in uint8/float; returns uint8 grayscale.
    """
    if gray.ndim != 2:
        raise ValueError("Expected grayscale 2D array")

    if gray.dtype == np.uint8:
        return gray

    g = gray.astype(np.float32)
    # If already in [0,255] we keep; otherwise normalize
    gmin, gmax = float(g.min()), float(g.max())
    if gmax <= 1.5:  # common if gray in [0,1]
        g = g * 255.0
    g = np.clip(g, 0.0, 255.0)
    return g.astype(np.uint8)


def _gradients(gray_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns gx, gy, mag, ang (ang in [0,180)).
    """
    g = gray_u8.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) * (180.0 / np.pi)) % 180.0
    return gx, gy, mag, ang


def hog_dim(dim_window: int | None = None) -> int:
    """
    Computes expected HOG feature length for a square window.
    """
    d = int(dim_window if dim_window is not None else params.dim_window)
    cell = int(params.hog_cell_size)
    bins = int(params.hog_num_bins)
    bs = int(params.hog_block_size)

    n_cells = d // cell
    n_blocks = (n_cells - bs + 1) * (n_cells - bs + 1)
    block_vec_len = (bs * bs * bins)
    return int(n_blocks * block_vec_len)


def hog_like(gray: np.ndarray) -> np.ndarray:
    """
    HOG-like features:
    - unsigned orientations [0,180)
    - per-cell histograms with *soft binning* (vote into two neighboring bins)
    - block L2 normalization over (block_size x block_size) cells
    - concatenated blocks -> feature vector (float32)

    Expects patch size = params.dim_window x params.dim_window.
    """
    gray_u8 = _to_gray_u8(gray)

    H, W = gray_u8.shape
    if H != params.dim_window or W != params.dim_window:
        raise ValueError(f"Expected patch {params.dim_window}x{params.dim_window}, got {H}x{W}")

    cell = int(params.hog_cell_size)
    bins = int(params.hog_num_bins)
    bs = int(params.hog_block_size)
    bin_width = 180.0 / float(bins)

    gx, gy, mag, ang = _gradients(gray_u8)

    n_cells_y = H // cell
    n_cells_x = W // cell

    # hist per cell
    hist = np.zeros((n_cells_y, n_cells_x, bins), dtype=np.float32)

    # Soft binning:
    # bin_f in [0,bins) float, vote to floor and next bin based on fractional part
    bin_f = ang / bin_width
    b0 = np.floor(bin_f).astype(np.int32)
    frac = (bin_f - b0.astype(np.float32)).astype(np.float32)
    b0 = np.clip(b0, 0, bins - 1)
    b1 = (b0 + 1) % bins  # wrap around

    # Accumulate per cell
    for cy in range(n_cells_y):
        y0, y1 = cy * cell, (cy + 1) * cell
        for cx in range(n_cells_x):
            x0, x1 = cx * cell, (cx + 1) * cell

            m = mag[y0:y1, x0:x1].ravel()
            bb0 = b0[y0:y1, x0:x1].ravel()
            bb1 = b1[y0:y1, x0:x1].ravel()
            ff = frac[y0:y1, x0:x1].ravel()

            # vote weights
            w0 = (1.0 - ff) * m
            w1 = ff * m

            # bincount for speed
            hist[cy, cx, :] += np.bincount(bb0, weights=w0, minlength=bins).astype(np.float32)
            hist[cy, cx, :] += np.bincount(bb1, weights=w1, minlength=bins).astype(np.float32)

    # Block normalization
    eps = 1e-6
    blocks = []
    for by in range(n_cells_y - bs + 1):
        for bx in range(n_cells_x - bs + 1):
            block = hist[by:by + bs, bx:bx + bs, :].reshape(-1).astype(np.float32)
            norm = np.sqrt(np.sum(block * block) + eps * eps)
            blocks.append(block / norm)

    feat = np.concatenate(blocks, axis=0).astype(np.float32)

    # Sanity: ensure expected dim
    expected = hog_dim(params.dim_window)
    if feat.size != expected:
        raise RuntimeError(f"HOG dim mismatch: got {feat.size}, expected {expected}")

    return feat


# -----------------------------
# Debug / visualization helpers
# -----------------------------
def hog_cell_energy(gray: np.ndarray) -> np.ndarray:
    """
    Per-cell gradient energy map (sum of magnitudes).
    Shape: (n_cells_y, n_cells_x). Great for debugging.
    """
    gray_u8 = _to_gray_u8(gray)
    _, _, mag, _ = _gradients(gray_u8)

    cell = int(params.hog_cell_size)
    H, W = gray_u8.shape
    ncy = H // cell
    ncx = W // cell

    E = np.zeros((ncy, ncx), dtype=np.float32)
    for cy in range(ncy):
        y0, y1 = cy * cell, (cy + 1) * cell
        for cx in range(ncx):
            x0, x1 = cx * cell, (cx + 1) * cell
            E[cy, cx] = float(mag[y0:y1, x0:x1].sum())
    return E


def visualize_hog(gray: np.ndarray, mode: str = "dominant") -> np.ndarray:
    """
    Visualization modes:
    - mode="dominant": draws dominant orientation per cell (simple, can look repetitive)
    - mode="energy": returns a heatmap-like BGR visualization of per-cell energy

    Returns BGR uint8 image suitable for cv2.imwrite.
    """
    gray_u8 = _to_gray_u8(gray)

    if mode == "energy":
        E = hog_cell_energy(gray_u8)
        # normalize to 0..255 and upscale to image size
        E_norm = E - E.min()
        if E_norm.max() > 1e-6:
            E_norm = E_norm / E_norm.max()
        heat = (E_norm * 255.0).astype(np.uint8)
        heat = cv2.resize(heat, (gray_u8.shape[1], gray_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
        heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        return heat_bgr

    # dominant orientation drawing
    cell = int(params.hog_cell_size)
    bins = int(params.hog_num_bins)
    bin_width = 180.0 / float(bins)

    _, _, mag, ang = _gradients(gray_u8)

    H, W = gray_u8.shape
    n_cells_y = H // cell
    n_cells_x = W // cell

    canvas = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0, y1 = cy * cell, (cy + 1) * cell
            x0, x1 = cx * cell, (cx + 1) * cell

            cell_mag = mag[y0:y1, x0:x1].ravel()
            cell_ang = ang[y0:y1, x0:x1].ravel()

            # full histogram (soft not needed for viz)
            hist = np.zeros((bins,), dtype=np.float32)
            b = np.floor(cell_ang / bin_width).astype(np.int32)
            b = np.clip(b, 0, bins - 1)
            hist += np.bincount(b, weights=cell_mag, minlength=bins).astype(np.float32)

            dom = int(np.argmax(hist))
            strength = float(hist[dom])
            if strength <= 1e-6:
                continue

            theta = (dom + 0.5) * bin_width * (np.pi / 180.0)
            cxp = int(x0 + cell / 2)
            cyp = int(y0 + cell / 2)

            # scale line length by relative strength within this cell
            L = int(min(cell / 2, 10))
            dx = int(np.cos(theta) * L)
            dy = int(np.sin(theta) * L)

            cv2.line(canvas, (cxp - dx, cyp - dy), (cxp + dx, cyp + dy), (0, 255, 0), 1)

    return canvas
