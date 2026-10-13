"""Small assertion helpers shared by the drill tests."""
import ast
import inspect
import textwrap

import numpy as np


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _dotted_names(obj):
    """Every name and dotted attribute the code uses (docstrings and comments ignored)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            parts = [node.attr]
            base = node.value
            while isinstance(base, ast.Attribute):
                parts.append(base.attr)
                base = base.value
            if isinstance(base, ast.Name):
                parts.append(base.id)
            found.add(".".join(reversed(parts)))
            found.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            found.update(f"{node.module}.{a.name}" for a in node.names)
            found.update(a.name for a in node.names)
    return found


def forbid(obj, names):
    """Fail if the drill's code uses a library function it must write itself.

    A dotted name ("cv2.add") must match exactly; a bare name ("filter2D")
    matches any attribute or name with that spelling.
    """
    used_names = _dotted_names(obj)
    used = [n for n in names if n in used_names]
    check(not used, f"this drill is 'from scratch': don't use {', '.join(used)}")


def arr(x, what):
    check(isinstance(x, np.ndarray), f"{what} must be a numpy array, got {type(x).__name__}")
    return x


def same_shape(a, b, what):
    check(a.shape == b.shape, f"{what}: shape {a.shape}, expected {b.shape}")


def close(got, want, tol, what):
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    check(got.shape == want.shape, f"{what}: shape {got.shape}, expected {want.shape}")
    err = np.abs(got - want)
    worst = np.unravel_index(np.argmax(err), err.shape) if err.size else ()
    check(err.size == 0 or err.max() <= tol,
          f"{what}: max error {err.max():.4g} > {tol} at index {tuple(int(i) for i in worst)} "
          f"(got {got[worst]:.4g}, expected {want[worst]:.4g})")
