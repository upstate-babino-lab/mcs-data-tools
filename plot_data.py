import os
os.environ["QT_QPA_PLATFORM"] = "xcb" # Avoid wayland errors
import matplotlib.pyplot as plt
import numpy as np


def plot_audio_data(raw, squared=None, smoothed=None, peaks=None):

    x = np.arange(0, len(raw))

    plt.figure(figsize=(12, 4))
    plt.plot(x, raw, label="Raw", alpha=0.4)
    if squared is not None:
        plt.plot(x, squared, label="Squared", alpha=0.4)
    if smoothed is not None:
        plt.plot(x, smoothed, label="Smoothed")
    if peaks is not None:
        plt.plot(peaks, smoothed[peaks], "rx", markersize=15, label="Peaks")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
