import math

import numpy as np
from cava_practice.testing import check, forbid


def test(m):
    forbid(m, ["sklearn", "fitLine"])
    for p, r, k in [(0.99, 0.5, 2), (0.99, 0.3, 2), (0.95, 0.5, 4), (0.999, 0.8, 3)]:
        want = math.ceil(math.log(1 - p) / math.log(1 - r ** k))
        got = m.ransac_iterations(p, r, k)
        check(got == want, f"ransac_iterations({p}, {r}, {k}) = {got}, expected {want}")
    rng = np.random.default_rng(7)
    xs = rng.uniform(0, 100, 60)
    inliers = np.c_[xs, 0.5 * xs + 10 + rng.normal(0, 0.4, 60)]
    outliers = rng.uniform(0, 100, (60, 2))
    pts = np.vstack([inliers, outliers])
    (a, b, c), mask = m.fit_line_ransac(pts, thresh=1.5, n_iter=200, seed=0)
    check(abs(a * a + b * b - 1) < 1e-6, "normalise the line so that a^2 + b^2 = 1")
    # true line: 0.5x - y + 10 = 0, normalised
    t = np.array([0.5, -1.0, 10.0]) / np.hypot(0.5, 1.0)
    got = np.array([a, b, c])
    if got @ t < 0:
        got = -got
    check(np.allclose(got[:2], t[:2], atol=0.02) and abs(got[2] - t[2]) < 0.5, f"line {got.round(3)} is not y = 0.5x + 10 (expected {t.round(3)})")
    mask = np.asarray(mask, bool)
    check(mask[:60].mean() > 0.95, f"only {mask[:60].mean():.0%} of the true inliers were marked as inliers")
    check(mask[60:].mean() < 0.15, f"{mask[60:].mean():.0%} of the outliers were marked as inliers")
