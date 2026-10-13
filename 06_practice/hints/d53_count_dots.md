- Crop a margin to drop the border, threshold (Otsu, inverted), open to
  kill specks, then cv2.connectedComponentsWithStats and keep components
  whose area is in a plausible range.
- Or cv2.HoughCircles with a radius range tied to the square size.
- Or findContours + circularity filter (4*pi*A/P^2 close to 1).
Make every size threshold RELATIVE to the square size: test squares vary.
