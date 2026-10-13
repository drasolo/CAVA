import numpy as np
from cava_practice import faces
from cava_practice.testing import check

BAR = 0.85


def test(m):
    results = {}
    for exam, load in faces.ALL.items():
        try:
            train, test = load()
        except FileNotFoundError as e:
            check(False, f"{exam}: {e}")
        Xtr = np.array([m.features(f) for f, _ in train])
        Xte = np.array([m.features(f) for f, _ in test])
        check(Xtr.ndim == 2, f"{exam}: features() must return 1D vectors of the same length")
        ytr = np.array([l for _, l in train])
        yte = np.array([l for _, l in test])
        clf = m.make_classifier().fit(Xtr, ytr)
        results[exam] = float((clf.predict(Xte) == yte).mean())
    print("    " + "  ".join(f"{k}: {v:.1%}" for k, v in results.items()))
    bad = {k: v for k, v in results.items() if v < BAR}
    check(not bad, f"below {BAR:.0%} on: " + ", ".join(f"{k} ({v:.1%})" for k, v in bad.items()))
