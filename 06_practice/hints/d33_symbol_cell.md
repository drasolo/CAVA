(pick one, then try another)
1. Drop a margin (~10-15%) to remove grid lines. Threshold (Otsu, inverted
   so ink = 255). Empty if there is almost no ink, OR if the contrast between
   the two Otsu classes is small (Otsu ALWAYS splits, even pure noise).
2. 0 has a HOLE, X doesn't: cv2.findContours with RETR_CCOMP and look for a
   child contour of reasonable area.
3. Or: the centre of a 0 is paper, the centre of an X is ink.
4. Or: circularity 4*pi*area/perimeter^2 of the largest contour.
