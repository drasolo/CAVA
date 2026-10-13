"""Scoring rules of the past lab exams, used by mocks/grade.py and by the
self-check cell in the mock notebooks.

All Ex2 scoring is all-or-nothing per image, as in the statements.
"""
from pathlib import Path


def _lines(path):
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def read_xo(path):
    """X&0 annotation: first line = number of filled cells, then 'pos symbol'."""
    lines = _lines(path)
    n = int(lines[0])
    cells = {}
    for l in lines[1:1 + n]:
        pos, sym = l.split()[:2]
        cells[int(pos)] = sym
    return n, cells


def read_single(path):
    return _lines(path)[0]


def read_ring(path):
    out = {}
    for l in _lines(path):
        pos, val = l.split()[:2]
        out[int(pos)] = val
    return out


def score_xo(pred_path, gt_path):
    """-> (positions_ok, symbols_ok, note)"""
    try:
        pn, pc = read_xo(pred_path)
    except Exception as e:
        return False, False, f"unreadable output ({e})"
    gn, gc = read_xo(gt_path)
    pos_ok = pn == gn and set(pc) == set(gc)
    sym_ok = pos_ok and all(pc[k] == gc[k] for k in gc)
    note = ""
    if pos_ok and not sym_ok and any(v in ("O", "o") for v in pc.values()):
        note = "you wrote the letter O; the annotations use the digit 0"
    elif not pos_ok:
        note = f"cells {sorted(pc)} vs true {sorted(gc)}"
    elif not sym_ok:
        note = f"symbols {pc} vs true {gc}"
    return pos_ok, sym_ok, note


def score_single(pred_path, gt_path):
    try:
        p = read_single(pred_path)
    except Exception as e:
        return False, f"unreadable output ({e})"
    g = read_single(gt_path)
    return p == g, "" if p == g else f"{p} vs true {g}"


def score_ring(pred1, pred2, gt1, gt2):
    """-> (stage1_ok, stage2_ok, note)"""
    try:
        p1, p2 = read_ring(pred1), read_ring(pred2)
    except Exception as e:
        return False, False, f"unreadable output ({e})"
    g1, g2 = read_ring(gt1), read_ring(gt2)
    s1 = p1 == g1
    s2 = p2 == g2
    bad = sorted(k for k in g2 if p2.get(k) != g2[k])
    return s1, s2, "" if s1 and s2 else f"wrong squares (stage 2): {bad}"
