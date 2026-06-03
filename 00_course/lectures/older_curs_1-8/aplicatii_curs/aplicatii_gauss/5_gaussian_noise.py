import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your image
img = cv2.imread("test.png")
if img is None:
    raise FileNotFoundError("⚠️ Could not find 'test.png'")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0  # normalize 0-1

# Create Gaussian noise
mean = 0
std = 0.1  # try 0.05, 0.1, 0.2
noise = np.random.normal(mean, std, img.shape)

# Add noise to image
noisy = np.clip(img + noise, 0, 1)

# Denoise with Gaussian filter
denoised = cv2.GaussianBlur(noisy, (5, 5), 1)

plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy)
plt.title("With Gaussian Noise")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(denoised)
plt.title("After Gaussian Blur")
plt.axis("off")

plt.tight_layout()
plt.show()
