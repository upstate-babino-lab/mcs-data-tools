#!/usr/bin/env python
#
# With a .h5 file exported from Multi Channel Systems DataManager
# extract the audio data from the analog-in-1 channel, and locate
# the precise timestamp of each tone.
#

import argparse
import h5py
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from h5_tools import get_date_and_duration, valid_filename
from plot_data import plot_audio_data

ANALOG_1_GROUP = "/Data/Recording_0/AnalogStream/Stream_3"


def get_audio_data(file_path):
    with h5py.File(file_path, "r") as f:
        group = f[ANALOG_1_GROUP]

        labelRaw = group.attrs.get("Label", "")
        if isinstance(labelRaw, bytes):
            labelStr = labelRaw.decode("utf-8")

        if not "Analog Data1" in labelStr:
            raise Exception(
                f"Expected string 'Analog Data1' not found in {ANALOG_1_GROUP} label {labelStr}"
            )

        if not "ChannelData" in group:
            raise Exception(
                f"Expected 'ChannelData' dataset not found in group {ANALOG_1_GROUP}"
            )

        info = group["InfoChannel"][()]
        microseconds_between_samples = info["Tick"][0]
        data_rate = round(1_000_000 / microseconds_between_samples)
        audio_data = group["ChannelData"][()].reshape(-1, 1)
        duration_seconds = len(audio_data) / data_rate
        print(
            f"{len(audio_data)} samples @{data_rate}hz"
            + f" duration={round(duration_seconds)} seconds ={duration_seconds / 60:.2f} minutes"
        )
        return audio_data


def main():
    parser = argparse.ArgumentParser(description="Process a single HDF5 (.h5) file.")
    parser.add_argument("filename", type=valid_filename, help="Path to the .h5 file")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot start of data")

    args = parser.parse_args()
    get_date_and_duration(args.filename)

    audio_data = get_audio_data(args.filename)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(audio_data)
    centered = audio_data - np.mean(audio_data)
    squared = scaler.fit_transform(centered**2)
    smoothed = scaler.fit_transform(
        savgol_filter((centered**2).flatten(), 1800, 3).reshape(audio_data.shape)
    )
    peaks, _ = find_peaks(smoothed.flatten(), height=0.5, distance=2000)
    if args.plot:
        min = 4000
        max = 7000
        peaks_in_range = peaks[(peaks > min) & (peaks < max)]
        plot_audio_data(
            scaled[min:max], squared[min:max], smoothed[min:max], peaks_in_range - min
        )
    print(f"Found {len(peaks)} peaks")
    diffs = 10_000 - np.diff(peaks)
    mean_value = np.mean(diffs)
    min_value = np.min(diffs)
    max_value = np.max(diffs)
    std_dev = np.std(diffs)

    # print(f"Diffs: {diffs}")
    print(
        f"Diffs mean: {mean_value:.2f} min: {min_value} max: {max_value} stdDev: {std_dev:.2f}"
    )

    print(peaks / 10_000)  # Timestamps in seconds


if __name__ == "__main__":
    main()
