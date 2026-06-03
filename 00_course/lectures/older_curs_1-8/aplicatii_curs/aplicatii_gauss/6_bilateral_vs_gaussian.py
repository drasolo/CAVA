import cv2
import matplotlib.pyplot as plt
import numpy as np

# === STEP 1: Load the image ===
img = cv2.imread("test.png")
if img is None:
    raise FileNotFoundError("⚠️ Could not find 'test.png'")

# OpenCV reads images in BGR (Blue-Green-Red) color order,
# but matplotlib (used for plotting) expects RGB order,
# so we convert it.
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# === STEP 2: Convert to grayscale for plotting intensity values ===
# A grayscale image has only shades of gray (no color),
# where each pixel is a number from 0 (black) to 255 (white).
# This makes it easier to show how pixel "brightness" changes.
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# === STEP 3: Define Gaussian blur strengths and colors ===
# These are the sigma (σ) values — how "wide" the Gaussian curve is.
# Small σ = light blur, large σ = heavy blur.
sigmas = [1, 3, 5, 10]

# Colors to make each σ easy to follow in the plots.
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# === STEP 4: Pick a horizontal line (a row of pixels) to analyze ===
# Images are grids of pixels: rows × columns.
# gray.shape[0] = number of rows (height)
# gray.shape[1] = number of columns (width)
# We pick the middle row to analyze how pixel values change across it.
row = gray.shape[0] // 2
x = np.arange(gray.shape[1])  # x = horizontal pixel positions

# === Create the figure with 3 parts ===
plt.figure(figsize=(13, 10))
plt.suptitle("Understanding Gaussian Blur: σ, Distribution & Intensity Profiles",
             fontsize=14, y=0.98)

# === PART 1: Show blurred images for each σ ===
for i, (s, c) in enumerate(zip(sigmas, colors), 1):
    # Apply Gaussian blur with standard deviation σ = s
    # (0, 0) tells OpenCV to pick kernel size automatically based on σ.
    blurred = cv2.GaussianBlur(img, (0, 0), s)

    # Display the blurred image in the top row
    plt.subplot(3, len(sigmas), i)
    plt.imshow(blurred)
    plt.title(f"σ = {s}", color=c)
    plt.axis("off")

# === PART 2: Show the Gaussian distribution curves ===
# These curves show the shape of the *weights* used for each blur strength.
plt.subplot(3, 1, 2)

# xg = distances from the center pixel (in “pixels”)
xg = np.linspace(-15, 15, 300)

for s, c in zip(sigmas, colors):
    # Formula for the 1D Gaussian function:
    # This gives us how important each neighbor pixel is.
    g = (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-(xg ** 2) / (2 * s ** 2))

    # Normalize: divide by the maximum so all curves fit in the same graph
    g /= g.max()

    # Plot the Gaussian curve
    plt.plot(xg, g, label=f"σ={s}", color=c, linewidth=2)

# The X-axis (distance) = how far a neighbor is from the pixel we are blurring.
# The Y-axis (relative weight) = how much that neighbor "matters".
# Closer pixels = higher weight (more influence), farther pixels = smaller weight.
plt.title("Gaussian Distribution Curves for Different σ Values")
plt.xlabel("Distance from center pixel (x)")
plt.ylabel("Relative weight (importance of neighbor)")
plt.legend()
plt.grid(True)

# === PART 3: Show pixel intensity profiles across the selected row ===
# "Pixel intensity" = brightness value (0 black → 255 white)
plt.subplot(3, 1, 3)

# Plot the original, unblurred intensity profile in black
plt.plot(x, gray[row, :], label="Original", color="black", linewidth=2)

# For each blur strength, plot the new smoothed line
for s, c in zip(sigmas, colors):
    # Apply Gaussian blur to the grayscale image
    blurred_gray = cv2.GaussianBlur(gray, (0, 0), s)
    # Plot the brightness values along the same row
    plt.plot(x, blurred_gray[row, :], label=f"σ={s}", color=c, linewidth=1.8)

# Here:
# - The X-axis = pixel position across the image (left to right)
# - The Y-axis = brightness of each pixel (0–255)
# When σ increases, the curve becomes smoother — the image gets blurrier.
plt.title(f"Pixel Intensity (Brightness) Across Row {row}")
plt.xlabel("Horizontal pixel position →")
plt.ylabel("Intensity (0–255)")
plt.legend()
plt.grid(True)

# Adjust layout so everything fits nicely
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
