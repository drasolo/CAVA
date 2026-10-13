"""HOG (histogram of oriented gradients) from scratch

Course: Curs 8 (Dalal-Triggs), Lab 9-10 (face detection), Tema 2, exam Ex1.
The feature you'll use most. Writing it once removes the mystery around
pixels_per_cell, cells_per_block and orientations.

TASK (match skimage.feature.hog exactly)
  cell_histograms(gray, ppc, nbins) -> array (n_cells_y, n_cells_x, nbins)
      1. gradients by central differences on a float64 image:
           g_row[y, x] = I[y+1, x] - I[y-1, x],  g_col[y, x] = I[y, x+1] - I[y, x-1]
           (0 on the first/last row for g_row, first/last column for g_col)
      2. magnitude = hypot(g_col, g_row); orientation = degrees(arctan2(g_row, g_col)) % 180
         (UNSIGNED: a dark->light edge and a light->dark edge vote the same).
      3. Bin i covers [i*180/nbins, (i+1)*180/nbins). Each cell (ppc x ppc pixels,
         cells start at the top-left, leftover pixels at the right/bottom are
         ignored) sums the magnitudes per bin, then divides by ppc*ppc.
  hog_descriptor(gray, ppc, cpb, nbins) -> 1D feature vector
      Blocks of cpb x cpb cells, sliding by ONE cell. Each block is flattened
      (in (cell_y, cell_x, bin) order) and L2-normalised: v / sqrt(sum(v^2) + 1e-10).
      Concatenate the blocks row by row.

RULES  no skimage.feature.hog, cv2.HOGDescriptor.
CHECK  Once it passes: why does HOG need every window resized to the same size?
"""


def cell_histograms(gray, ppc, nbins):
    raise NotImplementedError


def hog_descriptor(gray, ppc, cpb, nbins):
    raise NotImplementedError
