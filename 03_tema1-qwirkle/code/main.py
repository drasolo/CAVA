from config import settings
from calibrator import BoardCalibrator
from templates import TemplateExtractor
from detector import TileDetector
from classifier import TileClassifier
from validator import MoveValidator
from scorer import ScoreCalculator
from writer import SolutionWriter
from utils import load_image

import os
import numpy as np
import matplotlib.pyplot as plt  
import cv2                       

def main():
    print("=== Qwirkle Connect Pipeline Starting ===")

    # Take grid size from .env → config.py
    GRID_SIZE = settings["cell_count"]
    print(f"[CONFIG] GRID_SIZE from .env (CELL_COUNT) = {GRID_SIZE}")

    # ----------------------------------------------------
    # 1. Calibrate Board and Grid Coordinates
    # ----------------------------------------------------
    calibrator = BoardCalibrator(
        settings["empty_board_image"],
        grid_size=GRID_SIZE
    )

    if settings["debug"]:
        # --- BASIC SANITY CHECKS ---
        print("[CALIB] board_corners:\n", calibrator.board_corners)
        print("[CALIB] cell_size:", calibrator.cell_size)
        print("[CALIB] warped_size:", calibrator.warped_size)
        print("[CALIB] homography shape:", calibrator.homography.shape)

        # --- INTERNAL GEOMETRY SELF-TEST ---
        test_cells = [
            (0, 0),
            (0, GRID_SIZE - 1),
            (GRID_SIZE - 1, 0),
            (GRID_SIZE - 1, GRID_SIZE - 1),
            (GRID_SIZE // 2, GRID_SIZE // 2),
        ]

        print("[CALIB] Testing board_to_image ↔ image_to_board consistency...")
        for (r, c) in test_cells:
            x, y = calibrator.board_to_image(r, c)
            r2, c2 = calibrator.image_to_board(x, y)
            print(f"  cell ({r},{c}) -> warped ({x},{y}) -> back ({r2},{c2})")

        print("[CALIB] Testing homography round-trip (warped -> original -> board)...")
        H_inv = calibrator.inv_homography

        def warped_to_original(xw, yw):
            v = np.array([xw, yw, 1.0], dtype=np.float32)
            xo, yo, w = H_inv @ v
            xo /= w
            yo /= w
            return xo, yo

        for (r, c) in test_cells:
            xw, yw = calibrator.board_to_image(r, c)
            xo, yo = warped_to_original(xw, yw)
            r2, c2 = calibrator.original_image_to_board(xo, yo)
            print(f"  cell ({r},{c}) -> warped ({xw:.1f},{yw:.1f}) "
                  f"-> orig ({xo:.1f},{yo:.1f}) -> back ({r2},{c2})")

        # --- SAVE GRID IMAGE ---
        debug_path = "debug_board_grid.png"
        calibrator.save_debug_grid(debug_path)
        print(f"[CALIB] Saved calibration debug image to {debug_path}")

        # --- VISUAL CONFIRMATION WITH MATPLOTLIB ---
        # 1) Original empty board + detected corners
        orig_bgr = calibrator.image.copy()
        for (x, y) in calibrator.board_corners:
            cv2.circle(orig_bgr, (int(x), int(y)), 10, (0, 0, 255), 3)
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

        # 2) Warped board + grid overlay (we reuse the saved debug image)
        warped_bgr = cv2.imread(debug_path)
        if warped_bgr is not None:
            warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
        else:
            warped_rgb = None
            print("[WARN] Could not load debug grid image for visualization")

        # Show side by side
        if warped_rgb is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(orig_rgb)
            axes[0].set_title("Original empty board + corners")
            axes[0].axis("off")

            axes[1].imshow(warped_rgb)
            axes[1].set_title("Warped board + 16x16 grid")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()

    if settings["debug"]:
        debug_path = "debug_board_grid.png"
        calibrator.save_debug_grid(debug_path)
        print(f"[CALIB] Saved calibration debug image to {debug_path}")


    # Optional: save a debug image to verify calibration
    try:
        calibrator.save_debug_grid("debug_board_grid.png")
        print(f"Saved calibration debug image with grid_size={GRID_SIZE}")
    except AttributeError:
        # In case save_debug_grid isn't implemented / is changed later
        pass


    # ----------------------------------------------------
    # 2. Extract Shape + Color Templates (Only Once)
    # ----------------------------------------------------
    # From the all-tiles image, extract:
    # - 6 shape contours (used with matchShapes)
    # - 6 HSV color centers
    #
    # DEPENDS ON:
    # - templates.py
    template_extractor = TemplateExtractor(settings["tiles_reference_image"])
    shape_templates = template_extractor.extract_shapes()
    color_templates = template_extractor.extract_colors()

    # ----------------------------------------------------
    # 3. Initialize Tile Detector and Classifier
    # ----------------------------------------------------
    # These will:
    # - Compare j00 → j01, j01 → j02, etc.
    # - Detect which cells have new tiles (via diff)
    # - Classify tile shape and color per cell
    #
    # DEPENDS ON:
    # - detector.py
    # - classifier.py
    tile_detector = TileDetector(calibrator)
    classifier = TileClassifier(shape_templates, color_templates)

    # ----------------------------------------------------
    # 4. Initialize Validator + Scorer + Writer
    # ----------------------------------------------------
    # Validator: checks for legal move structure
    # Scorer: counts main line + perpendiculars + bonuses
    # Writer: outputs .txt for each move
    #
    # DEPENDS ON:
    # - validator.py
    # - scorer.py
    # - writer.py
    validator = MoveValidator()
    scorer = ScoreCalculator(calibrator)
    writer = SolutionWriter(settings["predictions_folder"])

    # ----------------------------------------------------
    # 5. Process Each Game Move (j01 to j20)
    # ----------------------------------------------------
    # Load j00.jpg to establish initial board state
    # Then loop through j01.jpg to j20.jpg:
    # - Detect new tiles
    # - Classify each tile
    # - Validate move
    # - Score it
    # - Save output .txt
    game_id = 1  # we’ll generalize later

    # all games are directly inside data/antrenare
    input_folder = settings["input_game_folder"]

    first_frame_name = f"{game_id}_00.jpg"      # 1_00.jpg, 2_00.jpg, etc.
    first_frame_path = os.path.join(input_folder, first_frame_name)

    print("[DEBUG] CWD:", os.getcwd())
    print("[DEBUG] input_folder:", input_folder)
    print("[DEBUG] first_frame_path:", first_frame_path)

    previous_frame = load_image(first_frame_path)
    board_state = {}  # {(row, col): (shape_id, color_id)}

    ##TODO to implement processing all 5 games
    for move_num in range(1, 21):
        print(f">>> Processing Move {move_num:02d}")

        move_str = f"{game_id}_{move_num:02d}"
        current_img_name = f"{game_id}_{move_num:02d}.jpg"
        current_img_path = os.path.join(input_folder, current_img_name)
        
        print("[DEBUG] current_img_path:", current_img_path)
        current_frame = load_image(current_img_path)

        # TODO 5.1: Detect new tile cells
        # new_tiles = tile_detector.detect_new_tiles(previous_frame, current_frame)

        # TODO 5.2: Classify shape and color for each new tile
        # classified_tiles = classifier.classify(current_frame, new_tiles)

        # TODO 5.3: Validate move legality (Qwirkle rules)
        # validator.check(classified_tiles, board_state)

        # TODO 5.4: Compute score
        # score = scorer.compute(classified_tiles, board_state)

        # TODO 5.5: Update board state with new tiles
        # board_state.update(classified_tiles)

        # TODO 5.6: Write output to .txt
        # writer.write_move(move_str, classified_tiles, score)

        previous_frame = current_frame  # Move forward

    print("=== Finished Processing ===")


if __name__ == "__main__":
    main()
