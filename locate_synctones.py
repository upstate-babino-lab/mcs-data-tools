#!/usr/bin/env python
#
# With a .h5 file exported from Multi Channel Systems DataManager
# extract the audio data from the analog-in-1 channel, and locate
# the precise timestamp of each synctone.
#

import argparse
import h5py
import numpy as np
from scipy.signal import savgol_filter, find_peaks, resample
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from h5_tools import get_date_and_duration, valid_filename
from plot_data import plot_data
from utils import centered_moving_average, find_square_wave_steps

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
    peak_indices, _ = find_peaks(smoothed.flatten(), height=0.5, distance=2000)
    if do_plot:
        min = peak_indices[0] - 2000
        max = peak_indices[0] + 2000
        peaks_in_range = peak_indices[(peak_indices > min) & (peak_indices < max)]
        plot_data(
            [scaled, squared, smoothed],
            min,
            max,
            ["scaled", "squared", "smoothed"],
            peaks_in_range - min,
        )

    diffs = (10_000 - np.diff(peak_indices)) / 10  # Diffs in milliseconds
    mean_value = np.mean(diffs)
    min_value = np.min(diffs)
    max_value = np.max(diffs)
    std_dev = np.std(diffs)
    print(
        f"Millisecond diffs mean: {mean_value:.2f} min: {min_value} max: {max_value} stdDev: {std_dev:.2f}"
    )
    return peak_indices / 10_000  # Timestamps in seconds


def locate_pda_transitions(file_path, do_plot=False):
    pda_data, _ = get_analog_data(file_path, "pda")
    scaler = RobustScaler()
    scaled = scaler.fit_transform(pda_data)
    thresholded = np.where(scaled > 0.5, 1, 0)
    cma = centered_moving_average(thresholded.flatten() ** 2, 101)
    thresholded = np.where(cma > 0.5, 1, 0)

    step_indices = find_square_wave_steps(thresholded, 0.1)
    if do_plot:
        min = 0
        max = 30000
        steps_in_range = step_indices[(step_indices > min) & (step_indices < max)]
        plot_data(
            [scaled, cma, thresholded],
            min,
            max,
            ["scaled", "cma", "thresholded"],
            steps_in_range - min,
        )

    return (
        step_indices[1:][:-1] / 10_000
    )  # Timestamps in seconds (without first and last)


def main():
    parser = argparse.ArgumentParser(description="Process a single HDF5 (.h5) file.")
    parser.add_argument("filename", type=valid_filename, help="Path to the .h5 file")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot start of data")

    args = parser.parse_args()
    get_date_and_duration(args.filename)

    np.set_printoptions(suppress=True)  # Suppress scientific notation

    synctone_timestamps = locate_synctones(args.filename, args.plot)
    print(f"Found {len(synctone_timestamps)} synctones")
    print(synctone_timestamps)

    pda_timestamps = locate_pda_transitions(args.filename, args.plot)
    print(f"Found {len(pda_timestamps)} PDA transitions")
    print(pda_timestamps)

    if len(synctone_timestamps) != len(pda_timestamps):
        raise Exception(
            f"Number of synctones ({len(synctone_timestamps)}) does not match PDA transitions ({len(pda_timestamps)})"
        )

    diffs = (pda_timestamps - synctone_timestamps) * 1000
    mean_delay = np.mean(diffs) # Mean delay of video detection relative to audio
    median_delay = np.median(diffs)  # Median delay
    centered_diffs = diffs - mean_delay
    min_magnitude = np.min(centered_diffs)
    max_magnitude = np.max(centered_diffs)
    std_dev = np.std(centered_diffs)
    print(
        f"{len(diffs)} diffs "
        f"mean: {mean_delay:.2f} "
        f"median: {median_delay:.2f} "
        f"min: {min_magnitude:.2f} max: {max_magnitude:.2f} stdDev: {std_dev:.2f}"
    )


if __name__ == "__main__":
    main()
