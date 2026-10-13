import numpy as np
from cava_practice.testing import check


def test(m):
    lines = ["0001.jpg 44 28 286 254\n", "001.jpg 226 64 398 232 Tommy  \n", "\n", "  ", "0003.jpg 1 2 3 4 unknown"]
    got = m.parse_annotations(lines)
    check(isinstance(got, list) and len(got) == 3, f"expected 3 records (blank lines skipped), got {got!r}")
    check(got[0] == {"file": "0001.jpg", "box": (44, 28, 286, 254), "label": None}, f"record 0 wrong: {got[0]!r}")
    check(got[1]["label"] == "Tommy" and got[1]["box"] == (226, 64, 398, 232), f"record 1 wrong: {got[1]!r}")
    check(all(isinstance(v, int) for v in got[2]["box"]), "box coordinates must be ints")

    img = np.arange(20 * 30).reshape(20, 30)
    c = m.crop(img, (2, 3, 5, 9))
    check(c.shape == (7, 4), f"crop (2,3,5,9) should be 7 tall x 4 wide (inclusive), got shape {c.shape}")
    check(c[0, 0] == img[3, 2], "crop starts at the wrong pixel: remember img[y, x]")
    c = m.crop(img, (-5, -4, 3, 2))
    check(c.shape == (3, 4), f"box sticking out top-left should clip to 3x4, got {c.shape}")
    c = m.crop(img, (25, 15, 40, 40))
    check(c.shape == (5, 5), f"box sticking out bottom-right should clip to 5x5, got {c.shape}")
    rgb = np.zeros((10, 10, 3))
    check(m.crop(rgb, (0, 0, 4, 4)).shape == (5, 5, 3), "crop must keep the colour channels")
