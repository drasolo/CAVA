import numpy as np
from cava_practice.testing import check


def test(m):
    boxes = m.cell_boxes(300, 3)
    check(len(boxes) == 9, f"3x3 board needs 9 boxes, got {len(boxes)}")
    check(tuple(boxes[0]) == (0, 0, 100, 100), f"first cell should be (0, 0, 100, 100), got {boxes[0]}")
    check(tuple(boxes[1]) == (100, 0, 200, 100), f"second cell is to the RIGHT of the first (row by row), got {boxes[1]}")
    check(tuple(boxes[8]) == (200, 200, 300, 300), f"last cell should end at the board edge, got {boxes[8]}")
    b = m.cell_boxes(400, 4, margin=0.1)
    check(tuple(b[5]) == (110, 110, 190, 190), f"4x4 on 400px with margin 0.1: cell 5 should be (110, 110, 190, 190), got {b[5]}")
    b = m.cell_boxes(500, 16)
    check(len(b) == 256 and b[-1][2] == 500 and b[-1][3] == 500, "16x16 boxes must cover the whole board, even when side/n isn't an integer")

    rm = np.asarray(m.number_map(3, "row_major"))
    check(rm.tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]], f"row_major 3x3 wrong: {rm.tolist()}")
    ring = np.asarray(m.number_map(5, "ring_cw"))
    want = [[1, 2, 3, 4, 5], [16, 0, 0, 0, 6], [15, 0, 0, 0, 7], [14, 0, 0, 0, 8], [13, 12, 11, 10, 9]]
    check(ring.tolist() == want, f"ring_cw 5x5 wrong:\n{ring}\nexpected\n{np.array(want)}")

    check(m.qwirkle_pos(1, 7) == "2H", f"qwirkle_pos(1, 7) should be '2H', got {m.qwirkle_pos(1, 7)!r}")
    check(m.qwirkle_pos(15, 15) == "16P", "qwirkle_pos(15, 15) should be '16P'")
    for s, rc in [("2H", (1, 7)), ("16P", (15, 15)), ("10A", (9, 0))]:
        check(tuple(m.parse_qwirkle_pos(s)) == rc, f"parse_qwirkle_pos({s!r}) should be {rc}")
