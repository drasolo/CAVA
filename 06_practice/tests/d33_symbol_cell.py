from collections import Counter

from cava_practice import synth
from cava_practice.testing import check


def test(m):
    errors = Counter()
    examples = {}
    total = 0
    for kind in ("X", "0", "empty"):
        for seed in range(30):
            size = 70 + (seed * 7) % 60
            real_seed = 1000 + seed + 100 * len(kind)
            cell = synth.symbol_cell(kind, size, seed=real_seed)
            got = m.classify_cell(cell)
            total += 1
            if got != kind:
                errors[(kind, got)] += 1
                examples.setdefault((kind, got), f"synth.symbol_cell('{kind}', {size}, seed={real_seed})")
    acc = 1 - sum(errors.values()) / total
    detail = ", ".join(f"{k}->{g}: {n} (e.g. {examples[(k, g)]})" for (k, g), n in errors.items())
    check(acc >= 0.97, f"accuracy {acc:.1%} < 97%. Confusions: {detail}. "
          "Reproduce one with the call shown and look at it (cv2.imshow or matplotlib).")
