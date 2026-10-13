Convert once with cv2.cvtColor(..., cv2.COLOR_BGR2HSV). Decide white
first (low S, high V). Use the MEDIAN hue of the patch, not the mean:
the mean of 2 and 178 is 90 (green!). Check which hue ranges the
colours actually have by printing medians of a few patches.
