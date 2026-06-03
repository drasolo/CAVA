import numpy as np
import matplotlib.pyplot as plt

def plot_gaussian(mu=0, sigma=1):
    x = np.linspace(-10, 10, 200)
    f = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-((x - mu)**2) / (2*sigma**2)) #formula nenorocita
    return x, f

sigmas = [0.5, 1, 2, 4]
plt.figure()

for s in sigmas:
    x, f = plot_gaussian(0, s)
    plt.plot(x, f, label=f"σ={s}")

plt.title("How σ changes the Gaussian curve")
plt.xlabel("x (distance)")
plt.ylabel("f(x) (weight)")
plt.legend()
plt.grid(True)
plt.show()
