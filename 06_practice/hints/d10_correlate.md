np.pad(img, ..., mode="edge") then loop over the KERNEL entries (a few),
not over the pixels (many): add kernel[i, j] * shifted_slice.
