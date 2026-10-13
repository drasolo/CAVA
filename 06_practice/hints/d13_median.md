np.lib.stride_tricks.sliding_window_view(padded, (k, k)) gives every
neighbourhood at once; then np.median over the last two axes.
