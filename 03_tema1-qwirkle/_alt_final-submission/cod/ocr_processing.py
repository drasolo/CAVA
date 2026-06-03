import numpy as np
import os
import glob
from parameters import *

def detect_bounding_box(image):
    if image is None or image.size == 0:
        print("Invalid or empty image passed to detect_bounding_box.")
        return None

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    _, thresh = cv.threshold(enhanced_gray, 50, 255, cv.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    h, w = morph.shape
    margin_h = int(h * 0.1)
    margin_w = int(w * 0.1)
    cropped_morph = morph[margin_h:h - margin_h, margin_w:w - margin_w]

    contours, _ = cv.findContours(cropped_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours detected.")
        return None

    filtered_contours = [c for c in contours if cv.contourArea(c) > 100]
    if not filtered_contours:
        print("No valid contours after filtering.")
        return None

    x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf
    for contour in filtered_contours:
        x, y, w, h = cv.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    if x_max <= x_min or y_max <= y_min:
        print(f"Invalid bounding box dimensions: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
        return None

    return int(x_min + margin_w), int(y_min + margin_h), int(x_max - x_min), int(y_max - y_min)



def get_centered_crop(image, bbox, size=(120, 120), zoom_out_factor=1.1):
    if not bbox:
        print("Invalid bounding box. Skipping crop.")
        return None

    x, y, w, h = bbox
    center_x = x + w // 2
    center_y = y + h // 2

    # Calculate crop coordinates with zoom-out factor
    new_w = int(w * zoom_out_factor)
    new_h = int(h * zoom_out_factor)
    crop_x = max(0, center_x - new_w // 2)
    crop_y = max(0, center_y - new_h // 2)
    crop_x_end = min(image.shape[1], crop_x + new_w)
    crop_y_end = min(image.shape[0], crop_y + new_h)

    cropped = image[crop_y:crop_y_end, crop_x:crop_x_end]

    if cropped.size == 0:
        print(f"Empty cropped region: bbox={bbox}, size={size}")
        return None

    # Ensure cropped image is resized to the target size
    return cv.resize(cropped, size, interpolation=cv.INTER_AREA)


def classify_number(template, image):
    """
    Match a template against the given image and return the matching score.
    """
    # Resize the template if it is larger than the image
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        scale_factor = min(
            image.shape[0] / template.shape[0],
            image.shape[1] / template.shape[1]
        )
        template = cv.resize(template, (int(template.shape[1] * scale_factor),
                                        int(template.shape[0] * scale_factor)),
                             interpolation=cv.INTER_AREA)

    result = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv.minMaxLoc(result)
    return max_val

def process_and_classify(image, templates, size=(120, 120)):
    """
    Process an image to classify the detected number with enhanced pre-processing.
    """
    # Detect the bounding box of the number in the image
    bbox = detect_bounding_box(image)
    if bbox:
        # Crop and preprocess the image
        cropped_image = get_centered_crop(image, bbox, size)
        matches_and_scores = []

        for template, filename in templates:
            # Preprocess the template with a median filter for noise reduction
            preprocessed_template = cv.medianBlur(template, 3)  # Apply median filter with kernel size 3

            # Compare the preprocessed template with the cropped image
            score = classify_number(preprocessed_template, cropped_image)
            matches_and_scores.append((filename, score))

        # Sort matches by score in descending order
        matches_and_scores.sort(key=lambda x: x[1], reverse=True)

        return matches_and_scores
    return []

def load_templates(template_folder):
    """
    Load templates from a specified folder with optional pre-processing.
    """
    templates = []
    if not os.path.exists(template_folder):
        raise FileNotFoundError(f"The specified template folder does not exist: {template_folder}")

    for filename in os.listdir(template_folder):
        template_path = os.path.join(template_folder, filename)
        template = cv.imread(template_path, cv.IMREAD_COLOR)  # Ensure templates are read in color

        if template is not None:
            # Preprocess template with median filtering
            filtered_template = cv.medianBlur(template, 3)
            templates.append((filtered_template, filename))
        else:
            print(f"Warning: Could not load template {filename}. Skipping.")

    return templates



def save_warped_templates(input_folder, output_folder, size=(120, 120)):
    """
    Save preprocessed (warped) templates to an output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    templates = load_templates(input_folder)
    for template, filename in templates:
        bbox = detect_bounding_box(template)
        if bbox:
            cropped_template = get_centered_crop(template, bbox, size)
            output_path = os.path.join(output_folder, filename)
            cv.imwrite(output_path, cropped_template)

def save_warped_images(input_folder, output_folder, size=(120, 120)):
    """
    Save preprocessed (warped) images from an input folder to an output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))
    for image_path in image_paths:
        image = cv.imread(image_path)
        bbox = detect_bounding_box(image)
        if bbox:
            cropped_image = get_centered_crop(image, bbox, size)
            output_path = os.path.join(output_folder, os.path.basename(image_path))
            cv.imwrite(output_path, cropped_image)
