import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ocr_processing import get_centered_crop, detect_bounding_box, process_and_classify
from parameters import *



def parse_turns(file_path):
    
    turns = []
    with open(file_path, 'r') as f:
        for line in f:
            player, turn = line.strip().split()
            turns.append((player, int(turn)))
    return turns

def compare_and_extract_pieces(current_frame, previous_frame, output_folder, image_name, templates, board):
    
    max_diff = 0
    max_diff_cell = None
    grid_size = BOARD_SIZE

    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate cell boundaries
            x_start = j * CELL_SIZE
            y_start = i * CELL_SIZE
            x_end = x_start + CELL_SIZE
            y_end = y_start + CELL_SIZE

            # Extract the current and previous cell regions
            current_cell = current_frame[y_start:y_end, x_start:x_end]
            previous_cell = previous_frame[y_start:y_end, x_start:x_end]

            # Compute the difference between cells
            diff = cv.absdiff(current_cell, previous_cell)
            diff_sum = np.sum(diff)

            # Track the cell with the highest difference
            if diff_sum > max_diff:
                max_diff = diff_sum
                max_diff_cell = (x_start, y_start, x_end, y_end, i, j)

    if max_diff_cell:
        x_start, y_start, x_end, y_end, row, col = max_diff_cell
        piece_image = current_frame[y_start:y_end, x_start:x_end]

        # Detect and crop the newly added piece
        bbox = detect_bounding_box(piece_image)
        if bbox:
            cropped_piece = get_centered_crop(piece_image, bbox, size=(120, 120))

            # Determine the grid position
            col_letter = chr(ord('A') + col)
            row_number = row + 1
            position = f"{row_number}{col_letter}"

            # Classify the piece
            matches_and_scores = process_and_classify(cropped_piece, templates)
            if matches_and_scores:
                best_match, _ = matches_and_scores[0]
                piece_value = int(os.path.splitext(best_match)[0])
            else:
                piece_value = 0

            # Update the board and save classification results
            board[row][col] = piece_value
            output_path = os.path.join(output_folder, f"{image_name}.txt")
            with open(output_path, 'w') as f:
                f.write(f"{position} {piece_value}")

            return piece_value  # Return the classified piece value
    return 0  # Return 0 if no significant difference is found

# Save results for Task 1
def save_results(output_path, image_name, position, value):
    with open(output_path, "a") as f:
        f.write(f"{position} {value}\n")

def visualize_board(board, previous_frame, output_folder, image_path):
    output_file = "task1_results.txt"
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Convert board to NumPy array
    board = np.array(board)
    
    # Debugging: Check board shape
    # print("Board shape:", board.shape)
    # print("Board contents:")
    print(board)
    
    # Check if board has valid dimensions
    if board.ndim != 2 or board.shape[0] != 14 or board.shape[1] != 14:
        raise ValueError("Board must be a 14x14 grid.")
    
    # Plot the board
    fig, ax = plt.subplots()
    ax.matshow(board != "#", cmap="Greens") 
     # Mark filled cells with green
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] != "#":
                position = f"{i+1}{chr(ord('A') + j)}"
                value = board[i, j]
                ax.text(j, i, str(value), va="center", ha="center")
                
                # Create results folder
                # subfolder = os.path.normpath(os.path.join(output_folder, "results"))
                # print(f"Results folder path: {subfolder}")
                # print(f"Absolute Results folder path: {os.path.abspath(subfolder)}")

                # if not os.path.exists(subfolder):
                #     print(f"Creating folder: {subfolder}")
                #     os.makedirs(subfolder, exist_ok=True)

                # Save result
                file_name = f"{image_name}_{i+1:02}{chr(ord('A') + j)}.txt"
                # result_file_path = os.path.join(subfolder, file_name)
                # print(f"Writing to file: {result_file_path}")
                # with open(result_file_path, "w") as f:
                #     f.write(f"{position} {value}\n")
                # print(f"Saved: {result_file_path}")

                # Annotate and save image
                annotated_image = annotate_image(previous_frame, position, value)
                annotated_output_path = os.path.join(output_folder, f"annotated_{image_name}.png")
                cv.imwrite(annotated_output_path, annotated_image)
                # print(f"Annotated image saved: {annotated_output_path}")

    
    # print(f"Results folder path: {subfolder}")

    plt.show()

def annotate_image(image, position, value):
    """
    Annotate the image with position and value.
    """
    font = cv.FONT_HERSHEY_SIMPLEX
    text = f"{value} ({position})"
    x, y = 50, 50  # Coordinates for text placement (adjust as needed)
    annotated_image = image.copy()
    cv.putText(annotated_image, text, (x, y), font, 1, (255, 0, 0), 2, cv.LINE_AA)
    return annotated_image

# Example usage:
# Assuming `previous_frame`, `output_folder`, and `image_path` are defined:

# def annotate_image(image, position, value):
#     font = cv.FONT_HERSHEY_SIMPLEX
#     text = f"{value} ({position})"
#     x, y = 50, 50  # Coordinates for text placement (adjust as needed)
#     cv.putText(image, text, (x, y), font, 1, (255, 0, 0), 2, cv.LINE_AA)
#     return image