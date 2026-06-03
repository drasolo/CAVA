import cv2
import numpy as np
from utils import load_image

class BoardCalibrator:
    def __init__(self, empty_board_path, grid_size=16):
        # 1. Load image
        self.image = load_image(empty_board_path)
        self.grid_size = grid_size

        # 2. Fields
        self.board_corners = None      # 4 corner points in original image
        self.cell_size = None          # pixel size of 1 cell in warped space
        self.homography = None         # original -> warped
        self.inv_homography = None     # warped -> original
        self.warped_size = None        # (width, height) of warped board in pixels

        # 3. Detect outer board corners
        self.board_corners = self.find_board_corners()

        # 4. Decide warped size based on board size in the original image
        #    (make cells roughly square with a reasonable size).
        board_width = np.linalg.norm(self.board_corners[1] - self.board_corners[0])
        board_height = np.linalg.norm(self.board_corners[3] - self.board_corners[0])
        board_size_pixels = int(min(board_width, board_height))

        # we want a nice round cell size; clamp between 60 and 140 px
        self.cell_size = max(60, min(140, board_size_pixels // self.grid_size))
        self.warped_size = (self.grid_size * self.cell_size,
                            self.grid_size * self.cell_size)

        # 5. Homography
        self.homography = self.compute_homography(self.board_corners, self.warped_size)
        self.inv_homography = np.linalg.inv(self.homography)

    def find_board_corners(self):
        """
        Detect board corners by segmenting the green border in HSV,
        then finding the largest green contour and approximating it
        to a quadrilateral.
        """
        img = self.image

        # --- 1) HSV + green mask ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # these ranges may need small tweaks, but are a good start
        lower_green = np.array([40, 40, 40], dtype=np.uint8)
        upper_green = np.array([90, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # --- 2) Morphological cleanup to connect border segments ---
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # DEBUG: save mask so we can see if green selection is OK
        cv2.imwrite("debug_board_green_mask.png", mask)

        # --- 3) Find contours on the mask ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("No contours found on green mask for board detection")

        # take the largest green contour
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        board_contour = contours[0]

        # DEBUG: visualize chosen contour
        dbg = img.copy()
        cv2.drawContours(dbg, [board_contour], -1, (0, 0, 255), 3)
        cv2.imwrite("debug_board_contour.png", dbg)

        # --- 4) Approximate to polygon / fallback to minAreaRect ---
        peri = cv2.arcLength(board_contour, True)
        approx = cv2.approxPolyDP(board_contour, 0.02 * peri, True)

        if len(approx) != 4:
            # if contour is noisy, use a minimal rotated rectangle
            rect = cv2.minAreaRect(board_contour)
            box = cv2.boxPoints(rect)      # 4 points
            approx = box.astype(np.float32)

        pts = approx.reshape(4, 2).astype(np.float32)
        return self._order_points(pts)


    def _order_points(self, pts):
        """
        Orders 4 points as: top-left, top-right, bottom-right, bottom-left.
        """
        # sum and diff of coordinates
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        return np.array([tl, tr, br, bl], dtype=np.float32)



    def compute_homography(self, src_pts, warped_size):
        w, h = warped_size  # width, height in pixels
        dst_pts = np.array([
            [0,     0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0,     h - 1]
        ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return H


    def warp_image(self, image):
        """
        Applies the precomputed homography to align the input image so
        that the board is a straight grid of size warped_size.
        """
        return cv2.warpPerspective(image, self.homography, self.warped_size)

    def get_cell_coords(self, row, col):
        """
        Returns (x1, y1, x2, y2) bounding box for cell [row, col]
        in the WARPED image coordinates.
        """
        x1 = col * self.cell_size
        y1 = row * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        return (x1, y1, x2, y2)

    def image_to_board(self, x, y):
        """
        Convert WARPED pixel coordinates (x, y) into (row, col).
        Assumes you already called warp_image on the frame.
        """
        row = int(y // self.cell_size)
        col = int(x // self.cell_size)
        return (row, col)

    def board_to_image(self, row, col):
        """
        Convert board (row, col) to WARPED pixel center (x, y).
        """
        x = int((col + 0.5) * self.cell_size)
        y = int((row + 0.5) * self.cell_size)
        return (x, y)
    
    def original_image_to_board(self, x, y):
        """
        Convert ORIGINAL image coords (x, y) -> (row, col)
        by first applying homography.
        """
        src = np.array([[x, y]], dtype=np.float32)
        src = np.array([src])
        dst = cv2.perspectiveTransform(src, self.homography)[0][0]
        return self.image_to_board(dst[0], dst[1])
    
    def save_debug_grid(self, out_path="debug_board_grid.png"):
        """
        Saves a debug image with the warped board and drawn grid lines.
        """
        warped = self.warp_image(self.image.copy())
        h, w = warped.shape[:2]

        # vertical lines
        for c in range(self.grid_size + 1):
            x = int(c * self.cell_size)
            cv2.line(warped, (x, 0), (x, h-1), (0, 255, 0), 2)

        # horizontal lines
        for r in range(self.grid_size + 1):
            y = int(r * self.cell_size)
            cv2.line(warped, (0, y), (w-1, y), (0, 255, 0), 2)

        cv2.imwrite(out_path, warped)

