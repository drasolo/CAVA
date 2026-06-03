import cv2
import matplotlib.pyplot as plt
import numpy as np

# === Load image ===
img = cv2.imread("test.png")
if img is None:
    raise FileNotFoundError("⚠️ Could not find 'test.png'")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale for the graph part
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# === Define σ values ===
sigmas = [1, 3, 5, 10]

# Pick a horizontal line (e.g., middle of image)
row = gray.shape[0] // 2
x = np.arange(gray.shape[1])

plt.figure(figsize=(12, 8))

# === 1. Show blur strength per σ ===
for i, s in enumerate(sigmas, 1):
    blurred = cv2.GaussianBlur(img, (0, 0), s)
    plt.subplot(2, len(sigmas), i)
    plt.imshow(blurred)
    plt.title(f"σ = {s}")
    plt.axis("off")

# === 2. Show intensity profile per σ ===
plt.subplot(2, 1, 2)

# Original profile
plt.plot(x, gray[row, :], label="Original", color="black", linewidth=2)

for s in sigmas:
    blurred_gray = cv2.GaussianBlur(gray, (0, 0), s)
    plt.plot(x, blurred_gray[row, :], label=f"σ={s}", linewidth=1.5)

plt.title(f"Pixel intensity across row {row}")
plt.xlabel("Horizontal pixel position →")
plt.ylabel("Intensity (0–255)")
plt.legend()
plt.grid(True)

plt.suptitle("Effect of σ (Standard Deviation) on Gaussian Blur", fontsize=14)
plt.tight_layout()
plt.show()
