import numpy as np
from cava_practice.testing import check


def test(m):
    tr, te = m.split_blocks(220, 55, 40)
    check(len(tr) == 160 and len(te) == 60, f"Feb 2025: expected 160 train / 60 test, got {len(tr)} / {len(te)}")
    check(tr[:3] == [1, 2, 3] and 40 in tr and 41 in te and 56 in tr and 95 in tr and 96 in te and 220 in te,
          "Feb 2025: ids 1-40, 56-95, ... are train; 41-55, 96-110, ... are test")
    tr, te = m.split_blocks(200, 50, 40)
    check(len(tr) == 160 and 41 in te and 51 in tr and 91 in te and 200 in te, "Feb 2025b split wrong (blocks of 50)")

    r = np.random.default_rng(0)
    labels = np.repeat(["Andy", "Ora", "Louie", "Tommy"], 55)
    r.shuffle(labels)
    names = [f"{i:03d}.jpg" for i in range(1, 221)]
    tr, te = m.split_by_order(names, list(labels), 45, 10)
    check(len(tr) == 180 and len(te) == 40, f"Jun 2025: expected 180 / 40, got {len(tr)} / {len(te)}")
    for lab in ("Andy", "Tommy"):
        idx = [n for n, l in zip(names, labels) if l == lab]
        check(set(idx[:45]) <= set(tr) and set(idx[45:]) <= set(te),
              f"Jun 2025: for {lab}, the first 45 files IN ANNOTATION ORDER must be train")
    tr, te = m.split_by_order(names, list(labels), 40, 10)
    check(len(tr) == 160 and len(te) == 40, "files after n_train + n_test of a label must be unused")

    ranges = [(1, 55, "Daphne"), (56, 110, "Fred"), (111, 165, "Velma"), (166, 220, "Shaggy")]
    for f, want in [("0001.jpg", "Daphne"), ("0055.jpg", "Daphne"), ("0056.jpg", "Fred"), ("data/ex1/0220.jpg", "Shaggy")]:
        got = m.label_from_ranges(f, ranges)
        check(got == want, f"label_from_ranges({f!r}) = {got!r}, expected {want!r}")

    img = np.arange(100 * 120).reshape(100, 120)
    p1 = m.random_patches(img, 10, 40, np.random.default_rng(42))
    p2 = m.random_patches(img, 10, 40, np.random.default_rng(42))
    check(len(p1) == 10 and all(p.shape == (40, 40) for p in p1), "need 10 patches of 40x40 fully inside the image")
    check(all(np.array_equal(a, b) for a, b in zip(p1, p2)), "same seed must give the same patches")
    g = np.random.default_rng(42)
    x0, y0 = int(g.integers(0, 81)), int(g.integers(0, 61))
    check(np.array_equal(p1[0], img[y0:y0 + 40, x0:x0 + 40]), "draw x0 first, then y0, with rng.integers(0, W - size + 1)")
