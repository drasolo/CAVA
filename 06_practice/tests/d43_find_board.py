import numpy as np
from cava_practice import synth
from cava_practice.testing import check


def test(m):
    worst = []
    for seed in range(20):
        img, truth = synth.board_scene(seed=300 + seed)
        got = np.asarray(m.find_board(img), np.float64).reshape(-1, 2)
        check(got.shape == (4, 2), f"scene {300 + seed}: return 4 corners, got shape {got.shape}")
        err = np.linalg.norm(got - truth, axis=1).max()
        worst.append(err)
        check(err <= 6, f"scene synth.board_scene(seed={300 + seed}): a corner is {err:.1f} px off "
              f"(got {got.round(0).tolist()}, true {truth.round(0).tolist()} in TL, TR, BR, BL order)")
    img, truth = synth.board_scene(seed=300)
    flat = np.asarray(m.warp_board(img, truth, 300))
    check(flat.shape == (300, 300, 3), f"warp_board(side=300) shape {flat.shape}")
    # the synthetic board has dark grid lines at 1/3 and 2/3: check they land there
    col = flat[150, :, :].mean(axis=1)
    for k in (100, 200):
        check(col[k - 6:k + 7].min() < col[130:170].mean() - 60,
              f"after warping, a vertical grid line should be near x={k}: is the corner order right?")
