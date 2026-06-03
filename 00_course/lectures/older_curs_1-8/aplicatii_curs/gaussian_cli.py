import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_gaussian_1d(mu=0, sigma=1):
    x = np.linspace(mu - 5*sigma, mu + 5*sigma, 300)
    f = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-((x - mu)**2) / (2*sigma**2))

    plt.plot(x, f, color='blue')
    plt.title(f'1D Gaussian Distribution (μ={mu}, σ={sigma})')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.show()


def show_gaussian_kernel(size=5, sigma=1):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)

    plt.imshow(kernel, cmap='viridis')
    plt.title(f'2D Gaussian Kernel ({size}x{size}, σ={sigma})')
    plt.colorbar()
    plt.show()


def apply_gaussian_blur(image_path, ksize=9, sigma=2, compare=False):
    img = cv2.imread(image_path)
    if img is None:
        print(f"\n❌ Error: Could not read image '{image_path}'")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gaussian = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)

    if compare:
        mean = cv2.blur(img, (ksize, ksize))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title('Original')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mean)
        plt.title('Mean Blur')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(gaussian)
        plt.title(f'Gaussian Blur (σ={sigma})')
        plt.axis('off')

        plt.show()
    else:
        plt.imshow(gaussian)
        plt.title(f'Gaussian Blurred (σ={sigma})')
        plt.axis('off')
        plt.show()


def main():
    while True:
        print("\n=== Gaussian CLI ===")
        print("1️⃣  Plot 1D Gaussian curve")
        print("2️⃣  Show 2D Gaussian kernel")
        print("3️⃣  Apply Gaussian blur to an image")
        print("4️⃣  Compare Gaussian vs Mean blur")
        print("0️⃣  Exit")

        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            mu = float(input("Enter μ (mean): ") or 0)
            sigma = float(input("Enter σ (std deviation): ") or 1)
            print("This function tells us how likely it is to see a certain value of x given that the data is normally distributed.")
            plot_gaussian_1d(mu, sigma)

        elif choice == "2":
            size = int(input("Enter kernel size (odd number): ") or 7)
            sigma = float(input("Enter σ (std deviation): ") or 1.5)
            show_gaussian_kernel(size, sigma)

        elif choice == "3":
            path = input("Enter path to image (default: test.png): ").strip() or "test.png"
            ksize = int(input("Enter kernel size (odd number): ") or 9)
            sigma = float(input("Enter σ (std deviation): ") or 2)
            apply_gaussian_blur(path, ksize, sigma)

        elif choice == "4":
            path = input("Enter path to image: ").strip() or "test.png"
            ksize = int(input("Enter kernel size (odd number): ") or 9)
            sigma = float(input("Enter σ (std deviation): ") or 2)
            apply_gaussian_blur(path, ksize, sigma, compare=True)

        elif choice == "0":
            print("\n👋 Exiting Gaussian CLI.")
            sys.exit(0)

        else:
            print("⚠️ Invalid choice, please try again.")


if __name__ == "__main__":
    main()
