import cv2
import numpy as np


# -------------------------------------------------------------------
# Fixed HSV color centers (from your trackbar experiments)
#  H in [0,179], S,V in [0,255]
# -------------------------------------------------------------------
COLOR_CENTERS = {
    "B": np.array([138.0, 222.5, 205.0]),  # blue
    "G": np.array([69.5, 231.5, 171.5]),   # green
    "W": np.array([90.0,  15.5, 227.0]),   # white
    "Y": np.array([60.5, 187.0, 213.5]),   # yellow
    "O": np.array([7.5,  168.0, 199.0]),   # orange
    "R": np.array([125.5,165.5, 194.5]),   # red
}

# Optional: weight Hue a bit more than S,V when computing distance
COLOR_WEIGHTS = np.array([2.0, 1.0, 1.0], dtype=np.float32)


class TileClassifier:
    """
    Classifies tiles by shape + color given:
      - shape_templates: {shape_id: contour}   (from your manual masks)
      - color_templates: unused for now (we use COLOR_CENTERS instead)

    Assumes:
      - `warped_image` is the board already warped by BoardCalibrator.warp_image
      - grid_size = 16 (from .env CELL_COUNT)
      - new_tiles are (row, col) in [0..15]x[0..15] with row 0 = top.
    """

    def __init__(self, shape_templates, color_templates=None, grid_size=16):
        self.shape_templates = shape_templates
        self.grid_size = grid_size

    def _classify_color_from_hsv(self, hsv_mean):
        """
        hsv_mean: (H, S, V) in OpenCV HSV space (H in [0,180])
        returns one of 'R','O','Y','W','G','B'
        """
        H, S, V = hsv_mean
        H = float(H)
        S = float(S)
        V = float(V)

        # White: almost no saturation, very bright
        if S < 30 and V > 160:
            return 'W'

        # Now decide based on hue
        # Red is near 0 or near 180
        if H < 10 or H >= 170:
            return 'R'

        # Orange
        if 10 <= H < 25:
            return 'O'

        # Yellow
        if 25 <= H < 40:
            return 'Y'

        # Green
        if 40 <= H < 90:
            return 'G'

        # Everything else on your board is Blue
        return 'B'

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def _crop_cell(self, warped_image, row, col):
        """
        Crops a single board cell (row, col) from a WARPED image.
        Uses simple grid division, assuming calibrator.warp_image()
        was used and grid_size matches.
        """
        h, w = warped_image.shape[:2]
        cell_h = h / self.grid_size
        cell_w = w / self.grid_size

        y1 = int(row * cell_h)
        y2 = int((row + 1) * cell_h)
        x1 = int(col * cell_w)
        x2 = int((col + 1) * cell_w)

        return warped_image[y1:y2, x1:x2].copy()

    def _extract_symbol_mask_and_contour(self, cell_bgr):
        """
        From a cell patch, build a binary mask with:
            symbol = white, background = black
        and return (mask, contour).
        """
        gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)

        # Otsu threshold
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Ensure symbol is white, background black.
        # If more than half the pixels are white, we assume background is white.
        if np.mean(thresh == 255) > 0.5:
            thresh = cv2.bitwise_not(thresh)

        # Small morphology to clean noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, None

        h, w = thresh.shape[:2]

        # Prefer contours that do NOT touch the border
        inner_contours = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            touches_border = (
                x <= 0 or y <= 0 or x + cw >= w - 1 or y + ch >= h - 1
            )
            if not touches_border:
                inner_contours.append(cnt)

        candidates = inner_contours if inner_contours else contours
        symbol_cnt = max(candidates, key=cv2.contourArea)

        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [symbol_cnt], -1, 255, thickness=-1)

        return mask, symbol_cnt

    def _classify_shape(self, contour):
        """
        Match contour against shape_templates using cv2.matchShapes.
        Returns (shape_id, score).
        """
        best_id = None
        best_score = float("inf")

        for shape_id, tpl_contour in self.shape_templates.items():
            score = cv2.matchShapes(
                contour, tpl_contour, cv2.CONTOURS_MATCH_I1, 0.0
            )
            if score < best_score:
                best_score = score
                best_id = shape_id

        return best_id, best_score

    def _classify_color(self, cell_bgr, symbol_mask):
        """
        Classify color based on mean HSV of the symbol area,
        using simple HSV ranges instead of COLOR_CENTERS.
        Returns:
            (color_code, distance)  # distance is dummy for now
        """
        hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
        symbol_pixels = hsv[symbol_mask == 255]

        if symbol_pixels.size == 0:
            return None, None

        mean_hsv = np.mean(symbol_pixels, axis=0)  # (H, S, V)
        H, S, V = mean_hsv
        H = float(H)
        S = float(S)
        V = float(V)

        # DEBUG (optional – keep while tuning, then remove)
        # print(f"[DEBUG] mean_hsv={mean_hsv}")

        # ---- RULES ----
        # White: low saturation, high value (bright)
        if S < 30 and V > 160:
            return "W", 0.0

        # Red: around 0° or near 180° in OpenCV HSV (0–180)
        if H < 10 or H >= 170:
            return "R", 0.0

        # Orange
        if 10 <= H < 25:
            return "O", 0.0

        # Yellow
        if 25 <= H < 40:
            return "Y", 0.0

        # Green
        if 40 <= H < 90:
            return "G", 0.0

        # Everything else -> Blue
        return "B", 0.0


    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def classify(self, warped_image, new_tiles):
        """
        Args:
            warped_image: BGR image already warped to board coordinates.
            new_tiles: list of (row, col) with 0-based indices.

        Returns:
            dict[(row, col)] = (shape_id, color_code)
        """
        results = {}

        for (row, col) in new_tiles:
            cell = self._crop_cell(warped_image, row, col)

            symbol_mask, contour = self._extract_symbol_mask_and_contour(cell)
            if contour is None:
                continue

            shape_id, _ = self._classify_shape(contour)
            color_code, _ = self._classify_color(cell, symbol_mask)

            if shape_id is None or color_code is None:
                continue

            results[(row, col)] = (shape_id, color_code)

        return results
