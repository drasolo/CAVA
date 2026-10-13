"""Connected components (and why 4 vs 8 connectivity matters)

Course: Curs 6. Counting dots on dice, separating tiles, finding the biggest
blob: all of it is connected-component labelling. Two pixels touching only at
a corner are ONE component with 8-connectivity and TWO with 4-connectivity.

TASK
  label_components(binary, connectivity) -> (n, labels)
      binary: HxW, nonzero = foreground. connectivity: 4 or 8.
      labels: int array, 0 = background, components numbered 1..n.
  component_areas(labels, n) -> list of n pixel counts, for labels 1..n.

RULES  no cv2.connectedComponents*, skimage.measure, scipy.
HINT   Breadth-first search from every unlabelled foreground pixel with a
       collections.deque. Or union-find over a single raster scan.
"""


def label_components(binary, connectivity):
    raise NotImplementedError


def component_areas(labels, n):
    raise NotImplementedError
