import os
import glob
from tile_detection import *
from ocr_processing import *
from image_processing import *
from parameters import *

def process_image(image_path, previous_frame, OUTPUT_FOLDER, templates, board):
    
    print(f"Processing {image_path}")
    frame = cv.imread(image_path)
    warped_frame = process_frame(frame)
    if warped_frame is not None:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        if previous_frame is not None:
            score = compare_and_extract_pieces(
                warped_frame, previous_frame, OUTPUT_FOLDER, image_name, templates, board
            )
            return warped_frame, score
    return warped_frame, 0

def generate_output():
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    empty_board_warped = process_frame(EMPTY_BOARD)

    image_paths = glob.glob(os.path.join(INPUT_FOLDER, "*.jpg"))
    # print(TEMPLATE_FOLDER)
    # print(f"TEMPLATE_FOLDER: {repr(TEMPLATE_FOLDER)}")
    templates = load_templates(TEMPLATE_FOLDER)

    game_number = 1
    current_turn = 1
    current_player = None
    cumulative_score = 0
    starting_turn = 1
    previous_frame = empty_board_warped
    board = [["#"] * 14 for _ in range(14)]  
    first_round_processed = False

    output_file = "task1_results.txt"
    open(output_file, "w").close()  # Clear the file before appending new results

    turns_file_path = os.path.join(INPUT_FOLDER, f"{game_number}_turns.txt")
    turns = parse_turns(turns_file_path)

    for frame_count, image_path in enumerate(image_paths):
        if frame_count % 50 == 0 and frame_count > 0:
            # Write the scores to the output file for the previous game
            with open(os.path.join(OUTPUT_FOLDER, f"{game_number}_scores.txt"), 'a') as f:
                if current_player is not None and first_round_processed:
                    f.write(f"{current_player} {starting_turn} {cumulative_score}\n")
            game_number += 1
            current_turn = 1
            current_player = None
            cumulative_score = 0
            starting_turn = 1
            first_round_processed = False

            # Reset the base frame and board for the new game
            previous_frame = empty_board_warped
            board = [["#"] * 14 for _ in range(14)]

            # Parse the turns file for the new game
            turns_file_path = os.path.join(INPUT_FOLDER, f"{game_number}_turns.txt")
            if not os.path.exists(turns_file_path):
                break
            turns = parse_turns(turns_file_path)

            # Initialize the first player and starting turn
            current_player, starting_turn = turns[0]

        # Process the current image
        image_name = os.path.basename(image_path)
        previous_frame, score = process_image(
            image_path, previous_frame, OUTPUT_FOLDER, templates, board
        )

        # Extract tile position and value

        # Save the results for logging
       
        # Determine the current player based on the turns list
        for player, turn in turns:
            if current_turn == turn:
                if current_player is not None and first_round_processed:
                    with open(os.path.join(OUTPUT_FOLDER, f"{game_number}_scores.txt"), 'a') as f:
                        f.write(f"{current_player} {starting_turn} {cumulative_score}\n")
                # Reset cumulative score and update current player and starting turn
                cumulative_score = 0
                current_player = player
                starting_turn = turn
                first_round_processed = True

        # Update the score for the current player
        cumulative_score += score
        current_turn += 1

    # Write the scores for the last game
    with open(os.path.join(OUTPUT_FOLDER, f"{game_number}_scores.txt"), 'a') as f:
        if current_player is not None and first_round_processed:
            f.write(f"{current_player} {starting_turn} {cumulative_score}\n")

    # Visualize the final board state
    
    visualize_board(board, previous_frame, OUTPUT_FOLDER, image_path)



if __name__ == "__main__":
    generate_output()
