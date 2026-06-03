import os
import cv2

from config import settings
from calibrator import BoardCalibrator
from templates import ManualTemplateLoader
from classifier import TileClassifier
from utils import load_image


# ------------------------------------------------------------
# Ground-truth parsing
# ------------------------------------------------------------
def parse_gt_line(line):
    """
    Parses a line like '10D 4G' -> (row_idx, col_idx, shape_id, color_code)
    row/col are 0-based indices for our internal board.

    Here we assume GT rows are counted from TOP to BOTTOM,
    same as the warped image:
        GT row 1  -> internal row 0
        GT row 16 -> internal row 15
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) != 2:
        # Probably score line (like: "7") or junk → ignore
        return None

    pos_str, tile_str = parts

    # Position: <row><colLetter>, e.g. "10D"
    row_part = pos_str[:-1]
    col_letter = pos_str[-1]

    row_idx = int(row_part) - 1          # top-down, 1..16 -> 0..15
    col_idx = ord(col_letter.upper()) - ord("A")

    # Tile: <shapeId><colorCode>, e.g. "4G"
    shape_id = int(tile_str[:-1])
    color_code = tile_str[-1].upper()

    return row_idx, col_idx, shape_id, color_code


def load_ground_truth(move_id_str):
    """
    Loads GT for a given move, e.g. move_id_str="1_01" -> data/antrenare/1_01.txt
    Returns:
        new_tiles: list[(row_idx, col_idx)]
        labels: dict[(row_idx, col_idx)] = (shape_id, color_code)
    """
    gt_path = os.path.join(settings["input_game_folder"], f"{move_id_str}.txt")

    new_tiles = []
    labels = {}

    with open(gt_path, "r") as f:
        for line in f:
            parsed = parse_gt_line(line)
            if parsed is None:
                continue
            row_idx, col_idx, shape_id, color_code = parsed
            new_tiles.append((row_idx, col_idx))
            labels[(row_idx, col_idx)] = (shape_id, color_code)

    return new_tiles, labels


# ------------------------------------------------------------
# Main test harness
# ------------------------------------------------------------
def main():
    # ---------- 1. Build calibrator for the real board ----------
    grid_size = settings["cell_count"]
    empty_board_path = settings["empty_board_image"]

    print(f"[INFO] Using empty board image: {empty_board_path}")
    calibrator = BoardCalibrator(empty_board_path, grid_size=grid_size)

    # ---------- 2. Load manual templates (shapes) ----------
    # Absolute correct path to manual_templates
    manual_templates_dir = r"C:\Users\drago\Downloads\CAVA_2025_2026_NEW\CAVA-2025-TEMA1\data\imagini_auxiliare\manual_templates"


    print(f"[INFO] Loading manual templates from: {manual_templates_dir}")

    loader = ManualTemplateLoader(manual_templates_dir)
    shape_templates = loader.load_shapes()
    color_templates = loader.load_colors()  # not used, but we load for completeness

    print(f"[INFO] Shape templates: {list(shape_templates.keys())}")
    print(f"[INFO] Color templates: {list(color_templates.keys())}")

    # ---------- 3. Build classifier ----------
    classifier = TileClassifier(
        shape_templates=shape_templates,
        color_templates=color_templates,
        grid_size=grid_size,
    )

    # ---------- 4. Pick a move to test ----------
    game_id = 1
    move_num = 1  # you can change this later
    move_id_str = f"{game_id}_{move_num:02d}"

    img_path = os.path.join(settings["input_game_folder"], f"{move_id_str}.jpg")
    print(f"[INFO] Testing classifier on move: {move_id_str}")
    print(f"[INFO] Image path: {img_path}")

    image = load_image(img_path)

    # Warp image to board coordinates using the SAME calibrator as runtime
    warped = calibrator.warp_image(image)

    # ---------- 5. Load ground truth for this move ----------
    new_tiles, labels = load_ground_truth(move_id_str)
    print(f"[INFO] GT tiles (row,col): {new_tiles}")

    # ---------- 6. DEBUG: per-tile crops + masks + classification ----------
    debug_root = os.path.join("debug_classifier", move_id_str)
    cells_dir = os.path.join(debug_root, "cells")
    masks_dir = os.path.join(debug_root, "masks")
    os.makedirs(cells_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    total = len(new_tiles)
    shape_correct = 0
    color_correct = 0
    both_correct = 0

    print("\n=== Per-tile results ===")

    for (row, col) in new_tiles:
        gt_shape, gt_color = labels[(row, col)]

        # 1) crop the cell like the classifier does
        cell = classifier._crop_cell(warped, row, col)

        # 2) get mask + contour
        symbol_mask, contour = classifier._extract_symbol_mask_and_contour(cell)

        # 3) save debug images
        cell_path = os.path.join(cells_dir, f"cell_r{row}_c{col}.png")
        cv2.imwrite(cell_path, cell)

        if symbol_mask is not None:
            mask_path = os.path.join(masks_dir, f"mask_r{row}_c{col}.png")
            cv2.imwrite(mask_path, symbol_mask)
        else:
            mask_path = None

        # 4) classify shape + color using the same internals
        if contour is None:
            pred_shape, pred_color = None, None
        else:
            pred_shape, _ = classifier._classify_shape(contour)
            pred_color, _ = classifier._classify_color(cell, symbol_mask)

        if pred_shape is None or pred_color is None:
            print(
                f"Cell ({row+1}, {chr(col + ord('A'))}): "
                f"GT=({gt_shape}{gt_color}), PRED=NONE "
                f"[cell={cell_path}, mask={mask_path}]"
            )
            continue

        ok_shape = (pred_shape == gt_shape)
        ok_color = (pred_color == gt_color)

        if ok_shape:
            shape_correct += 1
        if ok_color:
            color_correct += 1
        if ok_shape and ok_color:
            both_correct += 1

        print(
            f"Cell ({row+1}, {chr(col + ord('A'))}): "
            f"GT=({gt_shape}{gt_color}), "
            f"PRED=({pred_shape}{pred_color}), "
            f"shape_ok={ok_shape}, color_ok={ok_color} "
            f"[cell={cell_path}, mask={mask_path}]"
        )

    print("\n=== Summary ===")
    if total > 0:
        print(f"Total tiles: {total}")
        print(f"Shape accuracy: {shape_correct}/{total} = {shape_correct/total:.2%}")
        print(f"Color accuracy: {color_correct}/{total} = {color_correct/total:.2%}")
        print(f"Both correct: {both_correct}/{total} = {both_correct/total:.2%}")
    else:
        print("No tiles in GT? Check your .txt format / path.")


if __name__ == "__main__":
    main()
