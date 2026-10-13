hist = np.bincount(gray.ravel(), minlength=256); use cumulative sums so
every candidate t costs O(1).
