import cv2 as cv
import numpy as np

def detect_edges(image):
    """
    Detect edges in an image using Canny edge detection and dilate to thicken edges.
    """
    edges = cv.Canny(image, 50, 150)
    dilation_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    thick_edges = cv.dilate(edges, dilation_kernel, iterations=4)
    return thick_edges


def find_largest_contour(edges):
    """
    Find the largest contour in an edge-detected image.
    """
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv.contourArea)
    return None


def get_perspective_transform(image, contour, width, height):
    """
    Apply a perspective transform to an image based on the largest contour.
    """
    epsilon = 0.02 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        ordered_pts = order_points(pts)
        src = np.float32(ordered_pts)
        dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        matrix = cv.getPerspectiveTransform(src, dst)
        warped = cv.warpPerspective(image, matrix, (width, height))
        return warped
    return None


def apply_mask(image, lower_hsv, upper_hsv):
    """
    Apply an HSV mask to an image.
    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)
    masked = cv.bitwise_and(image, image, mask=mask)
    return masked, mask


def zoom_image(image, zoom_factor):
    """
    Zoom into an image by a specified zoom factor.
    """
    height, width = image.shape[:2]
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    resized = cv.resize(image, (new_width, new_height))
    start_x = (new_width - width) // 2
    start_y = (new_height - height) // 2
    cropped = resized[start_y:start_y + height, start_x:start_x + width]
    return cropped


def process_frame(frame):
    """
    Process an input frame to detect the game board and extract the region of interest.
    """
    width, height = 2030, 2030  # Dimensions of the Mathable game board
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([70, 255, 255])

    # Apply mask to the frame
    masked_frame, mask = apply_mask(frame, lower_hsv, upper_hsv)

    # Detect edges in the masked frame
    edges = detect_edges(masked_frame)

    # Find the largest contour and warp the image
    max_contour = find_largest_contour(edges)
    if max_contour is None:
        print("No contours found.")
        return None

    warped = get_perspective_transform(frame, max_contour, width, height)
    if warped is not None:
        # Zoom into the warped image
        return zoom_image(warped, 1.3)
    else:
        print("Could not find a valid perspective transform.")
        return None



def order_points(pts):
    """
    Order points in a consistent way: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
