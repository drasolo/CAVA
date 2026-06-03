
import cv2 as cv

BOARD_SIZE = 14  # Mathable board is 14x14
CELL_SIZE = 145  # Size of each cell in pixels (adjust if needed)

EMPTY_BOARD = cv.imread(r"C:\Users\drago\Desktop\CAVA\solomon_dragos_tema1_final\cod\empty_board.jpg")  
INPUT_FOLDER = r"C:\Users\drago\Desktop\CAVA\solomon_dragos_tema1_final\cod\testare"
OUTPUT_FOLDER = r"C:\Users\drago\Desktop\CAVA\solomon_dragos_tema1_final\cod\evaluare\fisiere_solutie\output"
TEMPLATE_FOLDER = r"C:\Users\drago\Desktop\CAVA\solomon_dragos_tema1_final\cod\templates"

SPECIAL_TILES = {
    "3x": ["1A", "1G", "1H", "1N", "7A", "7N", "8A", "8N", "14A", "14G", "14H", "14N"],
    "2x": ["2B", "2M", "3C", "3L", "4D", "4K", "5E", "5J", "10E", "10J", "11D", "11K", "12C", "12L", "13B", "13M"],
    "division": ["2E", "2J", "5B", "5M", "10A", "10M", "13E", "13J"],
    "subtraction": ["3F", "3I", "6C", "6L", "9C", "9L", "12F", "12I"],
    "addition": ["4G", "5H", "7E", "7K", "8D", "8J", "10G", "11H"],
    "multiplication": ["4H", "5G", "7D", "7J", "8E", "8K", "10H", "11G"]
}