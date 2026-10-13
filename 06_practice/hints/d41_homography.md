For each pair (x, y) -> (u, v), two rows of A h = b:
  [x, y, 1, 0, 0, 0, -u*x, -u*y] . h = u
  [0, 0, 0, x, y, 1, -v*x, -v*y] . h = v
Solve with np.linalg.solve, append 1, reshape to 3x3.
