import cv2

# >>> HARD-CODE YOUR IMAGE HERE <<<
path = r"CAVA-2025-TEMA1\debug_templates\shapes\shape1_mask_r2_c4.png"

img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

# invert
inv = 255 - img

# overwrite
cv2.imwrite(path, inv)

print("Done.")
