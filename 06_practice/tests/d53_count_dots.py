from cava_practice import synth
from cava_practice.testing import check


def test(m):
    wrong = []
    for seed in range(60):
        n = seed % 6 + 1
        size = 90 + (seed * 13) % 110
        got = m.count_pips(synth.dice_face(n, size, seed=500 + seed))
        if got != n:
            wrong.append(f"synth.dice_face({n}, {size}, seed={500 + seed}) -> {got}")
    check(not wrong, f"{len(wrong)}/60 wrong, e.g. {wrong[:3]}")
