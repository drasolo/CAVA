import numpy as np
from cava_practice import synth
from cava_practice.testing import check, forbid


def same_line(a, b, rtol=2.0, ttol=2.0):
    (r1, t1), (r2, t2) = a, b
    if abs(t1 - t2) <= ttol and abs(r1 - r2) <= rtol:
        return True
    # (rho, theta) and (-rho, theta +- 180) are the same line
    return abs(abs(t1 - t2) - 180) <= ttol and abs(r1 + r2) <= rtol


def test(m):
    forbid(m, ["HoughLines", "HoughLinesP", "skimage"])
    edges = np.zeros((50, 80), np.uint8)
    edges[10, 5] = 255
    acc, rhos, thetas = m.hough_accumulator(edges)
    D = int(np.ceil(np.hypot(50, 80)))
    check(len(thetas) == 180 and thetas[0] == 0, f"thetas must be arange(0, 180, 1), got len {len(thetas)}")
    check(rhos[0] == -D and len(rhos) == 2 * D + 1, f"rhos must run from -{D} to {D}; got {rhos[0]}..{rhos[-1]}")
    check(acc.shape == (len(rhos), len(thetas)), f"acc shape {acc.shape}, expected ({len(rhos)}, {len(thetas)})")
    check(acc.sum() == 180, f"one edge pixel votes once per theta: acc should sum to 180, got {acc.sum()}")
    j0 = 0
    i0 = int(np.argmin(np.abs(rhos - 5)))
    check(acc[i0, j0] == 1, "pixel (x=5, y=10) must vote for rho=5 at theta=0 (rho = x cos t + y sin t)")

    truth = [(40.0, 0.0), (70.0, 90.0), (100.0, 45.0), (30.0, 120.0)]
    img = synth.line_edges(truth, size=(200, 160), noise_pts=300, seed=2)
    acc, rhos, thetas = m.hough_accumulator(img)
    found = m.top_lines(acc, rhos, thetas, k=4)
    check(len(found) == 4, f"top_lines(k=4) returned {len(found)} lines")
    for line in truth:
        check(any(same_line(line, f) for f in found), f"line rho={line[0]}, theta={line[1]} not found; you returned {found}")
