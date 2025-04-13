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

        if not "Analog Data" in labelStr:
            raise Exception(
                f"Expected string 'Analog Data' not found in {ANALOG_1_GROUP} label {labelStr}"
            )

        if not "ChannelData" in group:
            raise Exception(
                f"Expected 'ChannelData' dataset not found in group {ANALOG_1_GROUP}"
            )

        # If there are several analog channels, find the one with 'audio' in the label
        info = group["InfoChannel"][()]
        nAnalogChannels = info.shape[0]
        print(f"Number of analog channels = {nAnalogChannels}")
        audioChannelNumber = 0
        if nAnalogChannels > 0:
            for i in range(nAnalogChannels):
                if (
                    "left" in info["Label"][i].decode("utf-8").lower()
                    or "audio" in info["Label"][i].decode("utf-8").lower()
                ):
                    audioChannelNumber = i

        label = info["Label"][audioChannelNumber].decode("utf-8")
        print(f"Using channel {audioChannelNumber} with label = '{label}'")

        microseconds_between_samples = info["Tick"][audioChannelNumber]
        data_rate = round(1_000_000 / microseconds_between_samples)

        analog_data = group["ChannelData"][audioChannelNumber, :]
        print("Analog data shape:", analog_data.shape)
        duration_seconds = len(analog_data) / data_rate
        print(
            f"{len(analog_data)} samples @{data_rate}hz"
            + f" duration={round(duration_seconds)} seconds ={duration_seconds / 60:.2f} minutes"
        )
        return (
            analog_data.reshape(-1, 1),
            data_rate,
        )  # Reshape to single-column 2D array


def main():
    parser = argparse.ArgumentParser(description="Process a single HDF5 (.h5) file.")
    parser.add_argument("filename", type=valid_filename, help="Path to the .h5 file")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot start of data")

    args = parser.parse_args()
    get_date_and_duration(args.filename)

    audio_data, data_rate = get_audio_data(args.filename)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(audio_data)
    centered = audio_data - np.mean(audio_data)
    squared = scaler.fit_transform(centered**2)
    smoothed = scaler.fit_transform(
        savgol_filter((centered**2).flatten(), 1800, 3).reshape(audio_data.shape)
    )
    peaks, _ = find_peaks(smoothed.flatten(), height=0.5, distance=2000)
    if args.plot:
        min = peaks[0] - 1500
        max = peaks[0] + 1500
        peaks_in_range = peaks[(peaks > min) & (peaks < max)]
        plot_audio_data(
            scaled[min:max], squared[min:max], smoothed[min:max], peaks_in_range - min
        )
    print(f"Found {len(peaks)} peaks")
    diffs = (10_000 - np.diff(peaks)) / 10  # Diffs in milliseconds
    mean_value = np.mean(diffs)
    min_value = np.min(diffs)
    max_value = np.max(diffs)
    std_dev = np.std(diffs)

    # print(f"Diffs: {diffs}")
    print(
        f"Millisecond diffs mean: {mean_value:.2f} min: {min_value} max: {max_value} stdDev: {std_dev:.2f}"
    )

    print("SyncTone locations (seconds):")
    print(peaks / 10_000)  # Timestamps in seconds


if __name__ == "__main__":
    main()
