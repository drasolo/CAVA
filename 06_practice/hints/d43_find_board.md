The board is the largest bright region, BUT light clutter can touch
it and a plain brightness threshold merges the two into one contour.
Threshold on what makes the board unique (bright AND unsaturated, in
HSV), then clean the mask with morphology. approxPolyDP's epsilon is a
fraction of the perimeter (start at 0.02 * cv2.arcLength(c, True)). If
you don't get exactly 4 points, adjust epsilon in a loop, or fall back
to cv2.minAreaRect / the convex hull.
Debug by drawing your corners on the image and saving it.
