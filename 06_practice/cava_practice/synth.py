"""Synthetic test images with known answers.

Each generator takes a seed so a failing test can be reproduced exactly.
"""
import cv2
import numpy as np


def rng(seed):
    return np.random.default_rng(seed)


def random_image(h, w, channels=3, seed=0):
    shape = (h, w, channels) if channels > 1 else (h, w)
    return rng(seed).integers(0, 256, shape, dtype=np.uint8)


def smooth_image(h, w, seed=0):
    """Random but smooth grayscale image (more like a photo than noise)."""
    small = rng(seed).integers(0, 256, (h // 8 + 1, w // 8 + 1)).astype(np.uint8)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = rng(seed + 1).normal(0, 6, (h, w))
    return np.clip(big + noise, 0, 255).astype(np.uint8)


def symbol_cell(kind, size=100, seed=0):
    """Grayscale cell: light paper, dark ink. kind in {'X', '0', 'empty'}."""
    r = rng(seed)
    paper = int(r.integers(185, 235))
    img = np.full((size, size), paper, np.uint8)
    ink = int(r.integers(10, 70))
    thick = int(r.integers(max(2, size // 25), max(3, size // 10)))
    c = size // 2 + r.integers(-size // 12, size // 12 + 1, 2)
    rad = int(size * r.uniform(0.22, 0.33))
    if kind == "X":
        a = r.uniform(-0.25, 0.25)
        for ang in (np.pi / 4 + a, -np.pi / 4 + a):
            d = np.array([np.cos(ang), np.sin(ang)]) * rad
            p1 = tuple(int(v) for v in c - d)
            p2 = tuple(int(v) for v in c + d)
            cv2.line(img, p1, p2, ink, thick, cv2.LINE_AA)
    elif kind == "0":
        axes = (rad, int(rad * r.uniform(0.75, 1.15)))
        cv2.ellipse(img, tuple(int(v) for v in c), axes, r.uniform(0, 180), 0, 360, ink, thick, cv2.LINE_AA)
    noise = r.normal(0, 8, img.shape)
    # a faint grid line fragment at the border, like a real cropped cell
    if r.random() < 0.5:
        img[: max(1, size // 30), :] = ink + 40
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def dice_face(n, size=120, seed=0):
    """BGR square with n round pips (1..6), plus noise and a light border."""
    r = rng(seed)
    bg = np.array([int(r.integers(190, 240))] * 3)
    img = np.ones((size, size, 3), np.uint8) * bg.astype(np.uint8)
    layouts = {
        1: [(0.5, 0.5)],
        2: [(0.25, 0.25), (0.75, 0.75)],
        3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
        4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
        5: [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)],
        6: [(0.25, 0.22), (0.75, 0.22), (0.25, 0.5), (0.75, 0.5), (0.25, 0.78), (0.75, 0.78)],
    }
    rad = int(size * r.uniform(0.06, 0.085))
    for fx, fy in layouts[n]:
        jx, jy = r.normal(0, size * 0.015, 2)
        center = (int(fx * size + jx), int(fy * size + jy))
        cv2.circle(img, center, rad, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.rectangle(img, (1, 1), (size - 2, size - 2), (60, 60, 60), max(1, size // 40))
    noise = r.normal(0, 7, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def board_scene(seed=0, size=(640, 480), grid_n=3):
    """A light, gridded board on a cluttered dark table, seen in perspective.

    Returns (bgr_image, corners) with corners ordered TL, TR, BR, BL (float32).
    """
    r = rng(seed)
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (int(r.integers(40, 90)), int(r.integers(40, 90)), int(r.integers(40, 90)))
    for _ in range(25):  # clutter: small blobs and lines, all smaller than the board
        color = tuple(int(v) for v in r.integers(0, 255, 3))
        p = tuple(int(v) for v in (r.integers(0, w), r.integers(0, h)))
        if r.random() < 0.5:
            cv2.circle(img, p, int(r.integers(3, 18)), color, -1)
        else:
            q = (p[0] + int(r.integers(-60, 60)), p[1] + int(r.integers(-60, 60)))
            cv2.line(img, p, q, color, int(r.integers(1, 4)))
    while True:  # redraw until the whole board is inside the frame
        side = int(min(w, h) * r.uniform(0.5, 0.7))
        cx, cy = w / 2 + r.uniform(-40, 40), h / 2 + r.uniform(-30, 30)
        ang = r.uniform(-0.35, 0.35)
        base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64) * side / 2
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        pts = base @ rot.T + [cx, cy]
        pts += r.normal(0, side * 0.04, pts.shape)  # perspective-ish distortion
        if (pts.min(axis=0) >= 10).all() and (pts.max(axis=0) <= [w - 11, h - 11]).all():
            break
    corners = pts.astype(np.float32)
    S = 300
    board = np.full((S, S, 3), int(r.integers(200, 240)), np.uint8)
    for i in range(1, grid_n):
        k = i * S // grid_n
        cv2.line(board, (k, 0), (k, S), (30, 30, 30), 4)
        cv2.line(board, (0, k), (S, k), (30, 30, 30), 4)
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [S, 0], [S, S], [0, S]]), corners)
    warped = cv2.warpPerspective(board, M, (w, h))
    mask = cv2.warpPerspective(np.full((S, S), 255, np.uint8), M, (w, h))
    img[mask > 0] = warped[mask > 0]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.clip(img + r.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return img, corners


def line_edges(lines, size=(200, 200), noise_pts=0, seed=0):
    """Binary edge image (uint8 0/255) containing lines given as (rho, theta_deg)."""
    w, h = size
    img = np.zeros((h, w), np.uint8)
    for rho, theta in lines:
        t = np.deg2rad(theta)
        c, s = np.cos(t), np.sin(t)
        x0, y0 = c * rho, s * rho
        p1 = (int(round(x0 - 1000 * s)), int(round(y0 + 1000 * c)))
        p2 = (int(round(x0 + 1000 * s)), int(round(y0 - 1000 * c)))
        cv2.line(img, p1, p2, 255, 1)
    if noise_pts:
        r = rng(seed)
        ys = r.integers(0, h, noise_pts)
        xs = r.integers(0, w, noise_pts)
        img[ys, xs] = 255
    return img
