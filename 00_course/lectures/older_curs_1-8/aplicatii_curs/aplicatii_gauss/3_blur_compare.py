import cv2
import matplotlib.pyplot as plt

img = cv2.imread("test.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mean_blur = cv2.blur(img, (9, 9))
gauss_blur = cv2.GaussianBlur(img, (9, 9), 2)

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mean_blur)
plt.title("Mean Blur")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gauss_blur)
plt.title("Gaussian Blur (σ=2)")
plt.axis('off')

plt.show()
