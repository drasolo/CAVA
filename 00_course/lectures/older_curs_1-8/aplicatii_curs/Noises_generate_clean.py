import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Load image ===
img = cv2.imread("test.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting

# Utility function for displaying
def show(title, image):
    plt.figure(figsize=(5,5))
    plt.imshow(np.clip(image, 0, 255).astype(np.uint8))
    plt.title(title)
    plt.axis('off')
    plt.show()

show("Original", img)

# === 1️⃣ Add SALT & PEPPER noise ===
def add_salt_pepper(image, amount=0.02):
    noisy = image.copy()
    h, w, c = image.shape
    num_salt = int(amount * h * w / 2)
    num_pepper = num_salt

    # Salt (white)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 255

    # Pepper (black)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 0

    return noisy

sp_noisy = add_salt_pepper(img)
show("Salt & Pepper Noise", sp_noisy)

# === 2️⃣ Add IMPULSE noise (white only) ===
def add_impulse(image, amount=0.02):
    noisy = image.copy()
    h, w, c = image.shape
    num_impulse = int(amount * h * w)
    coords = [np.random.randint(0, i - 1, num_impulse) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 255
    return noisy

impulse_noisy = add_impulse(img)
show("Impulse Noise (White Only)", impulse_noisy)

# === 3️⃣ Add GAUSSIAN noise ===
def add_gaussian(image, mean=0, sigma=20):
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy = image + gaussian
    return np.clip(noisy, 0, 255).astype(np.uint8)

gaussian_noisy = add_gaussian(img)
show("Gaussian Noise", gaussian_noisy)

# === CLEANING ===

# ✅ Best for Salt & Pepper: Median filter
clean_sp = cv2.medianBlur(sp_noisy, 3)
show("Cleaned (Median Filter)", clean_sp)

# ✅ Best for Impulse: also Median
clean_impulse = cv2.medianBlur(impulse_noisy, 3)
show("Cleaned Impulse (Median Filter)", clean_impulse)

# ✅ Best for Gaussian: Gaussian Blur
clean_gaussian = cv2.GaussianBlur(gaussian_noisy, (5, 5), sigmaX=1.5)
show("Cleaned Gaussian (Gaussian Blur)", clean_gaussian)


def compare(*images, titles=None):
    plt.figure(figsize=(15,5))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i+1)
        plt.imshow(np.clip(img, 0, 255).astype(np.uint8))
        plt.title(titles[i] if titles else f"Image {i+1}")
        plt.axis('off')
    plt.show()

compare(sp_noisy, clean_sp, titles=["Noisy", "Median Filtered"])
