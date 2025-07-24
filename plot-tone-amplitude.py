import numpy as np
import matplotlib.pyplot as plt

def plot_sine_vs_sine_powers(length):
    """Plots a sine wave and its powers"""

    x = np.linspace(0, length, length)
    radians = x * (np.pi / length)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, np.sin(radians), label='sin(x)')
    plt.plot(x, np.sin(radians)**2, label='sin(x)')
    plt.plot(x, np.sin(radians)**3, label='sin(x)^3')
    plt.plot(x, np.sin(radians)**4, label='sin(x)^4')
    plt.plot(x, np.sin(radians)**5, label='sin(x)^5')
    plt.plot(x, np.sin(radians)**6, label='sin(x)^6')
    plt.plot(x, np.sin(radians)**7, label='sin(x)^7')
    plt.plot(x, np.sin(radians)**8, label='sin(x)^8')
    plt.plot(x, np.sin(radians)**9, label='sin(x)^9')

    plt.xlabel('milliseconds')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
plot_sine_vs_sine_powers(200) # creates a plot of 100 points.
