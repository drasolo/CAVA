import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size=9, sigma=2):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

kernel = gaussian_kernel(50, sigma=10)
plt.imshow(kernel, cmap='viridis')
plt.title("2D Gaussian Kernel (σ=3)")
plt.colorbar(label='weight')
plt.show()
