import os
from dotenv import load_dotenv

load_dotenv()

settings = {
    "empty_board_image": os.getenv("EMPTY_BOARD_IMAGE"),
    "tiles_reference_image": os.getenv("TILES_REFERENCE_IMAGE"),
    "input_game_folder": os.getenv("INPUT_GAME_FOLDER"),
    "ground_truth_folder": os.getenv("GROUND_TRUTH_FOLDER"),
    "ground_truth_bonus_folder": os.getenv("GROUND_TRUTH_BONUS_FOLDER"),
    "predictions_folder": os.getenv("PREDICTIONS_FOLDER"),
    "predictions_bonus_folder": os.getenv("PREDICTIONS_BONUS_FOLDER"),
    "cell_count": int(os.getenv("CELL_COUNT")),
    "debug": os.getenv("DEBUG", "False").lower() == "true",
}
