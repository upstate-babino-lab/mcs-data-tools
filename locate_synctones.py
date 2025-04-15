#!/usr/bin/env python
#
# With a .h5 file exported from Multi Channel Systems DataManager
# extract the audio data from the analog-in-1 channel, and locate
# the precise timestamp of each synctone.
#

import argparse
import h5py
import numpy as np
from scipy.signal import savgol_filter, find_peaks, hilbert
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from h5_tools import get_date_and_duration, valid_filename
from plot_data import plot_data

ANALOG_1_GROUP = "/Data/Recording_0/AnalogStream/Stream_3"


def get_analog_data(file_path, desiredLabelStr):
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

        # Find the channel with {desiredLabelStr} in the label
        info = group["InfoChannel"][()]
        nAnalogChannels = info.shape[0]
        print(f"Number of analog channels = {nAnalogChannels}")
        labeledChannelNumber = -1
        if nAnalogChannels > 0:
            for i in range(nAnalogChannels):
                if desiredLabelStr in info["Label"][i].decode("utf-8").lower():
                    labeledChannelNumber = i
        if labeledChannelNumber < 0:
            raise Exception(
                f"Unable to find channel with label '{desiredLabelStr}' in {ANALOG_1_GROUP}"
            )

        label = info["Label"][labeledChannelNumber].decode("utf-8")
        print(f"Using channel {labeledChannelNumber} with label = '{label}'")

        microseconds_between_samples = info["Tick"][labeledChannelNumber]
        data_rate = round(1_000_000 / microseconds_between_samples)

        analog_data = group["ChannelData"][labeledChannelNumber, :]
        print("Analog data shape:", analog_data.shape)
        duration_seconds = len(analog_data) / data_rate
        print(
            f"{len(analog_data)} samples @{data_rate}hz"
            + f" duration={round(duration_seconds)} seconds ={duration_seconds / 60:.2f} minutes"
        )
        return (
            analog_data.reshape(-1, 1),  # Reshape to single-column 2D array
            data_rate,
        )


def locate_synctones(file_path, do_plot=False):
    audio_data, _ = get_analog_data(file_path, "audio")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(audio_data)
    centered = audio_data - np.mean(audio_data)
    squared = scaler.fit_transform(centered**2)
    smoothed = scaler.fit_transform(
        savgol_filter((centered**2).flatten(), 1800, 3).reshape(audio_data.shape)
    )
    peaks, _ = find_peaks(smoothed.flatten(), height=0.5, distance=2000)
    if do_plot:
        min = peaks[0] - 2000
        max = peaks[0] + 2000
        peaks_in_range = peaks[(peaks > min) & (peaks < max)]
        plot_data(
            scaled[min:max], squared[min:max], smoothed[min:max], peaks_in_range - min
        )

    print(f"Found {len(peaks)} peaks")
    diffs = (10_000 - np.diff(peaks)) / 10  # Diffs in milliseconds
    mean_value = np.mean(diffs)
    min_value = np.min(diffs)
    max_value = np.max(diffs)
    std_dev = np.std(diffs)
    print(
        f"Millisecond diffs mean: {mean_value:.2f} min: {min_value} max: {max_value} stdDev: {std_dev:.2f}"
    )
    return peaks / 10_000  # Timestamps in seconds


def locate_pda_transitions(file_path, do_plot=False):
    pda_data, _ = get_analog_data(file_path, "pda")
    scaler = RobustScaler()
    scaled = scaler.fit_transform(pda_data)
    thresholded = np.where(scaled > 0.5, 1, 0)
    if do_plot:
        min = 1000
        max = 30000
        plot_data(scaled[min:max], thresholded[min:max])


def main():
    parser = argparse.ArgumentParser(description="Process a single HDF5 (.h5) file.")
    parser.add_argument("filename", type=valid_filename, help="Path to the .h5 file")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot start of data")

    args = parser.parse_args()
    get_date_and_duration(args.filename)

    synctone_timestamps = locate_synctones(args.filename, args.plot)
    print(synctone_timestamps)

    pda_timestamps = locate_pda_transitions(args.filename, args.plot)


if __name__ == "__main__":
    main()
