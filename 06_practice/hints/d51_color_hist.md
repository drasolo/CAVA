Compute each channel's bin index, then one combined index
(ih * bs + is) * bv + iv and np.bincount(..., minlength=...).
