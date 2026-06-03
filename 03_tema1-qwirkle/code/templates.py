import os
import cv2
import numpy as np
import glob


from calibrator import BoardCalibrator
from utils import load_image


# -------------------------------------------------------------------
# REFERENCE CELLS ON THE ALL-TILES BOARD
# -------------------------------------------------------------------
# These are 0-based (row, col) indices in the *board grid* of the
# all-tiles image (TILES_REFERENCE_IMAGE from .env).
#
# YOU NEED TO UPDATE THESE after you run the calibrator on the
# all-tiles image and inspect the warped grid.
#
# For example, if circles are on board row 2 and red column is 3
# in the warped all-tiles board, then:
#     SHAPE_REF_CELLS[1] = (2, 3)     # 1 = circle
#     COLOR_REF_CELLS["R"] = (2, 3)   # R = red
#
# For now they are placeholders so the file is valid Python.
# -------------------------------------------------------------------

SHAPE_REF_CELLS = {
    1: (2, 4),  # TODO: (row, col) for CIRCLE
    2: (4, 8),  # TODO: (row, col) for CLOVER
    3: (6, 8),  # TODO: (row, col) for DIAMOND
    4: (3, 8),  # TODO: (row, col) for SQUARE
    5: (5, 8),  # TODO: (row, col) for 4-POINT STAR
    6: (7, 9),  # TODO: (row, col) for 8-POINT STAR
}

COLOR_REF_CELLS = {
    "R": (2, 2),  # TODO: (row, col) for a RED tile
    "O": (6, 8),  # TODO: ORANGE
    "Y": (6, 12),  # TODO: YELLOW
    "W": (9, 4),  # TODO: WHITE
    "G": (9, 9),  # TODO: GREEN
    "B": (9, 12),  # TODO: BLUE
}


class TemplateExtractor:
    """
    Extracts shape and color templates from the 'all tiles' board image.

    Pipeline:
      - Run BoardCalibrator on the all-tiles image to get a warped 16x16 grid.
      - For each reference cell (row, col):
          * crop that cell
          * for shapes: threshold + largest contour
          * for colors: HSV + mean over symbol region
      - Optionally save debug crops so you can visually check templates.
    """

    def __init__(self, reference_image_path: str, grid_size: int = 16):
        self.reference_image_path = reference_image_path
        self.grid_size = grid_size

        # Calibrate using the all-tiles board instead of the empty board.
        # Same logic as in main.py, just with a different image. 
        self.calibrator = BoardCalibrator(reference_image_path, grid_size=grid_size)

        # Warped version where the board is a perfect grid.
        self.warped_ref = self.calibrator.warp_image(self.calibrator.image.copy())

        # Where we'll drop visual debug templates.
        self.debug_root = "CAVA-2025-TEMA1/debug_templates"
        os.makedirs(self.debug_root, exist_ok=True)
        os.makedirs(os.path.join(self.debug_root, "shapes"), exist_ok=True)
        os.makedirs(os.path.join(self.debug_root, "colors"), exist_ok=True)

        # Optional: save a grid overlay for the all-tiles board so you
        # can see which (row, col) corresponds to which tile.
        grid_path = os.path.join(self.debug_root, "all_tiles_grid.png")
        try:
            self.calibrator.save_debug_grid(grid_path)
        except Exception:
            # if save_debug_grid ever changes, we don't die here
            pass

    # ------------------------------------------------------------
    # Internal helper: crop a cell from the warped reference board
    # ------------------------------------------------------------
    def _crop_cell(self, row: int, col: int):
        x1, y1, x2, y2 = self.calibrator.get_cell_coords(row, col)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        return self.warped_ref[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------
    # 1. SHAPE TEMPLATES
    # ------------------------------------------------------------
    def extract_shapes(self):
        """
        Returns:
            dict[int, contour]  # {shape_id: contour}
        Also saves debug crops and masks inside debug_templates/shapes/.
        """
        shape_templates = {}

        for shape_id, rc in SHAPE_REF_CELLS.items():
            row, col = rc
            cell = self._crop_cell(row, col)

            # Save raw crop for sanity check
            debug_bgr_path = os.path.join(
                self.debug_root, "shapes", f"shape{shape_id}_cell_r{row}_c{col}.png"
            )
            cv2.imwrite(debug_bgr_path, cell)

            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

            # Otsu threshold: symbols are bright on dark background
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Ensure symbol is white and background is black:
            # if more than half of the pixels are white, we assume
            # the background is white and we invert the mask.
            if np.mean(thresh == 255) > 0.5:
                thresh = cv2.bitwise_not(thresh)


            # Small morphology to clean up noise
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

            debug_mask_path = os.path.join(
                self.debug_root, "shapes", f"shape{shape_id}_mask_r{row}_c{col}.png"
            )
            cv2.imwrite(debug_mask_path, thresh)

            

        return shape_templates

    # ------------------------------------------------------------
    # 2. COLOR TEMPLATES
    # ------------------------------------------------------------
    def extract_colors(self):
        """
        Returns:
            dict[str, np.ndarray]  # {color_code: mean_hsv_vec}
        Also saves debug crops and masks inside debug_templates/colors/.
        """
        color_templates = {}

        for code, rc in COLOR_REF_CELLS.items():
            row, col = rc
            cell = self._crop_cell(row, col)

            debug_bgr_path = os.path.join(
                self.debug_root, "colors", f"color_{code}_cell_r{row}_c{col}.png"
            )
            cv2.imwrite(debug_bgr_path, cell)

            hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)

            # Use V channel to separate bright symbol from dark tile.
            v = hsv[:, :, 2]
            _, mask = cv2.threshold(
                v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Ensure symbol is white and background is black
            if np.mean(mask == 255) > 0.5:
                mask = cv2.bitwise_not(mask)

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            debug_mask_path = os.path.join(
                self.debug_root, "colors", f"color_{code}_mask_r{row}_c{col}.png"
            )
            cv2.imwrite(debug_mask_path, mask)

            symbol_pixels = hsv[mask == 255]
            if symbol_pixels.size == 0:
                raise RuntimeError(
                    f"[TemplateExtractor] No symbol pixels for color '{code}' "
                    f"at cell (row={row}, col={col}). "
                    f"Check COLOR_REF_CELLS and the all-tiles image."
                )

            mean_hsv = np.mean(symbol_pixels, axis=0)  # (H, S, V)
            color_templates[code] = mean_hsv

        return color_templates
    

class ManualTemplateLoader:
    """
    Loads shape + color templates from manually prepared PNGs.

    Expected structure (base_dir is usually data/imagini_auxiliare/manual_templates):

        base_dir/
          shapes/
            shape1_cell_*.png
            shape1_mask_*.png
            ...
            shape6_cell_*.png
            shape6_mask_*.png

          colors/
            color_R_cell_*.png
            color_R_mask_*.png
            ...
            color_B_cell_*.png
            color_B_mask_*.png
    """

    def __init__(self, base_dir="data/imagini_auxiliare/manual_templates"):
        self.base_dir = base_dir
        self.shapes_dir = os.path.join(base_dir, "shapes")
        self.colors_dir = os.path.join(base_dir, "colors")

        if not os.path.isdir(self.shapes_dir):
            raise FileNotFoundError(f"Shapes dir not found: {self.shapes_dir}")
        if not os.path.isdir(self.colors_dir):
            raise FileNotFoundError(f"Colors dir not found: {self.colors_dir}")

    def _find_single(self, pattern: str):
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No file matches pattern: {pattern}")
        # If there are many, just take the first.
        return matches[0]

    def load_shapes(self):
        """
        Returns:
            dict[int, contour]  # {shape_id: contour}
        Uses only the *_mask_*.png files (white symbol, black background).
        """
        shape_templates = {}

        for shape_id in range(1, 7):
            mask_pattern = os.path.join(
                self.shapes_dir, f"shape{shape_id}_mask_*.png"
            )
            mask_path = self._find_single(mask_pattern)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read mask: {mask_path}")

            # Find contour of the symbol (white on black)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                raise RuntimeError(f"No contour found in {mask_path}")

            # Largest contour = symbol
            symbol_cnt = max(contours, key=cv2.contourArea)
            shape_templates[shape_id] = symbol_cnt

        return shape_templates

    def load_colors(self):
        """
        Returns:
            dict[str, np.ndarray]  # {color_code: mean_hsv_vec}
        Uses *_cell_*.png + *_mask_*.png for each color.
        """
        color_templates = {}
        color_codes = ["R", "O", "Y", "W", "G", "B"]

        for code in color_codes:
            cell_pattern = os.path.join(
                self.colors_dir, f"color_{code}_cell_*.png"
            )
            mask_pattern = os.path.join(
                self.colors_dir, f"color_{code}_mask_*.png"
            )

            cell_path = self._find_single(cell_pattern)
            mask_path = self._find_single(mask_pattern)

            cell_bgr = cv2.imread(cell_path)
            if cell_bgr is None:
                raise FileNotFoundError(f"Could not read color cell: {cell_path}")

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read color mask: {mask_path}")

            hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
            symbol_pixels = hsv[mask == 255]
            if symbol_pixels.size == 0:
                raise RuntimeError(
                    f"No symbol pixels in color '{code}' mask: {mask_path}"
                )

            mean_hsv = np.mean(symbol_pixels, axis=0)
            color_templates[code] = mean_hsv

        return color_templates

