numpy slices are [y, x], i.e. img[ymin:ymax+1, xmin:xmax+1].
Negative slice starts wrap around in numpy. Clip BEFORE slicing.
